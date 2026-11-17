"""
make.fun v1 harness: Forecasts hourly Twitch concurrency for Grand Theft Auto VI
using a log-linear comparable regression model with shrinkage and residual quantile mapping.
"""

import os
import sys
import json
from datetime import datetime
from math import log, exp, sqrt
from statistics import NormalDist

# Ensure the script's own directory is first on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def round2(val):
    """Round half up to 2 decimal places; return int if whole number."""
    val = float(val)
    # JavaScript style Math.round(x * 100) / 100 for positive numbers
    rounded = int(val * 100 + 0.5) / 100.0
    if rounded.is_integer():
        return int(rounded)
    return rounded

def parse_iso(dt_str):
    """Parse ISO-8601 string safely."""
    return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))

def run():
    input_dir = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
    output_dir = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')
    os.makedirs(output_dir, exist_ok=True)

    # Process every *.in file in name order
    if not os.path.exists(input_dir):
        return

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith('.in'):
            continue

        in_path = os.path.join(input_dir, fname)
        with open(in_path, 'r', encoding='utf-8') as f:
            case = json.load(f)

        levels = case.get('levels', [0.05, 0.25, 0.5, 0.75, 0.95])
        
        # Find known viewer for hour 1
        known_rows = case.get('known', [])
        p1 = None
        for r in known_rows:
            if r.get('hour') == 1 and r.get('viewers') is not None:
                p1 = float(r.get('viewers'))
                break
        if p1 is None:
            hh = case.get('history_hourly', [])
            if hh and hh[-1].get('viewers') is not None:
                p1 = float(hh[-1].get('viewers'))
        if p1 is None:
            h_list = case.get('history', [])
            if h_list and h_list[-1].get('viewers') is not None:
                p1 = float(h_list[-1].get('viewers'))
        if p1 is None or p1 <= 0:
            p1 = 1.0
        else:
            p1 = max(1.0, p1)

        # Filter pool of comparables
        comparables = case.get('comparables', [])
        pool = [c for c in comparables if 'midnight local' in c.get('unlock', '').lower()]
        if len(pool) < 8:
            pool = comparables

        # Build hourly maps for pool games
        pool_data = []
        for c in pool:
            try:
                ds = parse_iso(c['day_start'])
                hourly_map = {}
                for h_entry in c.get('hourly', []):
                    v = h_entry.get('viewers')
                    if v is not None and v > 0:
                        dt = parse_iso(h_entry['from'])
                        h_idx = int((dt - ds).total_seconds() // 3600) + 1
                        if h_idx not in hourly_map:
                            hourly_map[h_idx] = v
                if 1 in hourly_map:
                    pool_data.append((hourly_map[1], hourly_map))
            except Exception:
                continue

        # House-baseline standard normal quantiles
        normal_dist = NormalDist()
        z_vals = [normal_dist.inv_cdf(tau) for tau in levels]

        hours_out = []
        log_p1 = log(p1)

        fc_meta = case.get('forecast', [])
        forecast_hours = [row.get('hour') for row in fc_meta if row.get('hour') is not None]
        if not forecast_hours:
            forecast_hours = list(range(2, 25))

        # Forecast hours
        for h in forecast_hours:
            points = []
            for p1_g, h_map in pool_data:
                if h in h_map:
                    points.append((log(p1_g), log(h_map[h] / p1_g)))

            n = len(points)
            if n < 6:
                # Fallback to house-baseline shape
                shape_std = 0.15 * sqrt(h - 1)
                q_list = [p1 * exp(z * shape_std) for z in z_vals]
            else:
                xs = [pt[0] for pt in points]
                ys = [pt[1] for pt in points]
                mean_x = sum(xs) / n
                mean_y = sum(ys) / n

                num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
                den = sum((xs[i] - mean_x) ** 2 for i in range(n))
                
                b_raw = (num / den) if den != 0 else 0.0
                # Shrink slope
                b = b_raw * n / (n + 5)
                a = mean_y - b * mean_x

                min_x, max_x = min(xs), max(xs)
                clamped_log_p1 = max(min_x, min(max_x, log_p1))
                m = a + b * clamped_log_p1
                # Clamp median log-ratio
                m = max(log(0.10), min(log(8.0), m))

                residuals = sorted([ys[i] - (a + b * xs[i]) for i in range(n)])

                q_list = []
                for tau in levels:
                    # Empirical quantile index via linear interpolation
                    idx = (n - 1) * tau
                    lower_idx = int(idx)
                    upper_idx = min(n - 1, lower_idx + 1)
                    weight = idx - lower_idx
                    resid_val = residuals[lower_idx] * (1 - weight) + residuals[upper_idx] * weight
                    q_val = p1 * exp(m + resid_val)
                    q_list.append(q_val)

            if len(q_list) >= 5 and q_list[0] > 0 and q_list[4] > 0:
                if log(q_list[4]) - log(q_list[0]) < 0.4:
                    med_log = log(q_list[2]) if q_list[2] > 0 else (log(q_list[0]) + log(q_list[4])) / 2.0
                    q_list[0] = exp(med_log - 0.2)
                    q_list[4] = exp(med_log + 0.2)

            # Enforce non-decreasing quantiles & clamp
            cleaned_q = []
            running_max = 0.01
            for val in q_list:
                val = max(0.01, min(1e9, val))
                if val < running_max:
                    val = running_max
                running_max = val
                cleaned_q.append(round2(val))

            hours_out.append({
                "hour": h,
                "q": cleaned_q
            })

        out_data = {"hours": hours_out}
        out_name = os.path.splitext(fname)[0] + '.out'
        out_path = os.path.join(output_dir, out_name)

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, separators=(',', ':'))
            f.write('\n')

if __name__ == '__main__':
    run()