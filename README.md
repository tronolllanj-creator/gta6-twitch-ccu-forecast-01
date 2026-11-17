# GTA VI Twitch CCU Forecast

Forecasts Grand Theft Auto VI's launch-day Twitch concurrent viewers hour-by-hour (hours 2-24) based on the first hour's print, outputting five quantiles (0.1, 0.3, 0.5, 0.7, 0.9) per hour.

## The Model
- **Core Engine:** Per-hour log-linear regression of $\log(\text{viewers}_h / \text{viewers}_1)$ on $\log(\text{hour-1 viewers})$ using historical midnight-local game launches.
- **Shrinkage:** Slope coefficients are shrunk toward zero by $n / (n + 5)$ to prevent overfitting on sparse historical data.
- **Uncertainty:** Prediction intervals are derived from empirical residual quantiles around the fitted conditional median.
- **Fallback:** Uses a house-baseline fallback if a comparable pool hour has fewer than 6 data points.

## Validation
- **Performance:** Achieves a leave-one-out pinball loss of **0.2396** on historical midnight-local launches (compared to **0.3568** for the house baseline).
- **Robustness:** All format and determinism checks pass successfully on the kit's 4 practice cases.

## Design
- Written in **pure Python standard library** (no external dependencies).
- Fully **deterministic** and runs completely **offline**.

## How to Run
Set the input and output directories via environment variables and execute the main script:

```bash
export MAKEFUN_INPUT_DIR=/path/to/inputs
export MAKEFUN_OUTPUT_DIR=/path/to/outputs
python main.py
```