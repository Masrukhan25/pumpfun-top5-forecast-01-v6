# pumpfun-top5-forecast-01 Entry

This program forecasts the top 5 new Solana coins by market cap, 24 hours out, for the `pumpfun-top5-forecast-01` bounty.

## How It Works

The forecasting model generates ranking probabilities via a Monte Carlo simulation of coin market caps over the target forecast horizons, incorporating a horizon-mixture size model and unnamed-share optimisation to derive order probabilities for the top rankings.

## How to Run

Execute the program with:

```bash
python3 main.py
```

- **Inputs:** Reads input data from the environment variable `$MAKEFUN_INPUT_DIR`.
- **Outputs:** Writes the output forecast to `$MAKEFUN_OUTPUT_DIR/forecast.json`.

## Requirements & Properties

- Uses only the Python standard library.
- Deterministic.

Local faithful backtest (10 reads, 4 splits): pooled log score -34.191.
