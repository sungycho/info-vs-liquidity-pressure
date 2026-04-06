# Method 0 Backtest Result

## Data
- Event file: `Methods_Sung_Cho/0_event_features.parquet`
- CRSP file: `data/processed/crsp_daily.parquet`
- Events: 4,181
- Event date range: 2023-01-05 to 2024-12-20
- Pressure score range: [-0.660, 0.618]
- NaN pressure scores: 0

## Quintile Backtest (Q5 - Q1)

### entry=t=0, horizon=5d
- Data availability: 4180/4181 (0.02% loss)
- Q1 return: 0.0052 (0.52%), Q5 return: 0.0024 (0.24%), Q5-Q1: -0.0029 (-0.29%)
- Monotonic Q1→Q5: No

| quintile | n_events | mean_return | std_return |
|---:|---:|---:|---:|
| 1 | 837 | 0.005246 | 0.082975 |
| 2 | 836 | -0.001420 | 0.074629 |
| 3 | 836 | 0.006284 | 0.082711 |
| 4 | 835 | 0.004174 | 0.077982 |
| 5 | 836 | 0.002384 | 0.073456 |

### entry=t=0, horizon=20d
- Data availability: 4147/4181 (0.81% loss)
- Q1 return: 0.0102 (1.02%), Q5 return: 0.0135 (1.35%), Q5-Q1: 0.0033 (0.33%)
- Monotonic Q1→Q5: No

| quintile | n_events | mean_return | std_return |
|---:|---:|---:|---:|
| 1 | 834 | 0.010174 | 0.117814 |
| 2 | 824 | 0.013901 | 0.100312 |
| 3 | 828 | 0.016665 | 0.098320 |
| 4 | 830 | 0.018654 | 0.107133 |
| 5 | 831 | 0.013479 | 0.098783 |

### entry=t=+1, horizon=5d
- Data availability: 4179/4181 (0.05% loss)
- Q1 return: 0.0036 (0.36%), Q5 return: 0.0014 (0.14%), Q5-Q1: -0.0022 (-0.22%)
- Monotonic Q1→Q5: No

| quintile | n_events | mean_return | std_return |
|---:|---:|---:|---:|
| 1 | 837 | 0.003630 | 0.065767 |
| 2 | 836 | -0.001540 | 0.061914 |
| 3 | 835 | 0.006064 | 0.062657 |
| 4 | 835 | 0.005389 | 0.061644 |
| 5 | 836 | 0.001381 | 0.059326 |

### entry=t=+1, horizon=20d
- Data availability: 4144/4181 (0.88% loss)
- Q1 return: 0.0093 (0.93%), Q5 return: 0.0121 (1.21%), Q5-Q1: 0.0028 (0.28%)
- Monotonic Q1→Q5: No

| quintile | n_events | mean_return | std_return |
|---:|---:|---:|---:|
| 1 | 834 | 0.009333 | 0.105379 |
| 2 | 823 | 0.014619 | 0.093264 |
| 3 | 828 | 0.015173 | 0.082302 |
| 4 | 829 | 0.018763 | 0.092574 |
| 5 | 830 | 0.012146 | 0.089378 |

## Extreme Bin Test (20d Horizon)

| top/bottom pct | entry | top_n | bottom_n | top_return | bottom_return | spread |
|---:|---|---:|---:|---:|---:|---:|
| 10% | t=0 | 417 | 417 | 0.0078 | 0.0094 | -0.0016 |
| 10% | t=+1 | 417 | 417 | 0.0102 | 0.0056 | 0.0046 |
| 15% | t=0 | 624 | 625 | 0.0116 | 0.0069 | 0.0047 |
| 15% | t=+1 | 623 | 625 | 0.0122 | 0.0048 | 0.0074 |
| 20% | t=0 | 831 | 833 | 0.0135 | 0.0103 | 0.0032 |
| 20% | t=+1 | 830 | 833 | 0.0121 | 0.0094 | 0.0028 |

## Interpretation
- The main 20-day long-short spread is positive under both entry conventions.
- 5-day spreads: t=0 `-0.0029`, t=+1 `-0.0022`. 20-day spreads: t=0 `0.0033`, t=+1 `0.0028`.
- If Q5-Q1 is persistently negative, the current pressure direction is likely inverted for return prediction.
- Saved event-level backtest outputs in `data/results/method0/` for follow-up diagnostics.

---

# Traditional Backtester

======================================================================
Event-Driven Backtester - Real Data Test
======================================================================

Loading event data: Methods_Sung_Cho/0_event_features.parquet
✓ Loaded 4181 rows
✓ Prepared 4181 unique events
  Date range: 2023-01-05 00:00:00 to 2024-12-20 00:00:00
  pressure_score range: [-0.660, 0.618]
  NaN scores: 0

Loading CRSP returns: data/processed/crsp_daily.parquet
✓ Loaded 271,569 daily return observations
  Stocks: 531
  Date range: 2022-12-06 to 2024-12-31
  Mean return: 0.00054

----------------------------------------------------------------------
Initializing EventBacktester...
----------------------------------------------------------------------

Running full analysis (4 configurations)...
======================================================================

Running: entry=t=0, horizon=5 days
  Data availability: 4180/4181 events (0.0% loss)
  By quintile (valid/total):
    Q1: 837/837
    Q2: 836/836
    Q3: 836/836
    Q4: 835/836
    Q5: 836/836
  Q1 return: 0.0052
  Q5 return: 0.0024
  Q5-Q1:     -0.0029

Running: entry=t=0, horizon=20 days
  Data availability: 4147/4181 events (0.8% loss)
  By quintile (valid/total):
    Q1: 834/837
    Q2: 824/836
    Q3: 828/836
    Q4: 830/836
    Q5: 831/836
  Q1 return: 0.0102
  Q5 return: 0.0135
  Q5-Q1:     0.0033

Running: entry=t=+1, horizon=5 days
  Data availability: 4179/4181 events (0.0% loss)
  By quintile (valid/total):
    Q1: 837/837
    Q2: 836/836
    Q3: 835/836
    Q4: 835/836
    Q5: 836/836
  Q1 return: 0.0036
  Q5 return: 0.0014
  Q5-Q1:     -0.0022

Running: entry=t=+1, horizon=20 days
  Data availability: 4144/4181 events (0.9% loss)
  By quintile (valid/total):
    Q1: 834/837
    Q2: 823/836
    Q3: 828/836
    Q4: 829/836
    Q5: 830/836
  Q1 return: 0.0093
  Q5 return: 0.0121
  Q5-Q1:     0.0028

======================================================================
RESULTS SUMMARY
======================================================================

──────────────────────────────────────────────────────────────────────
Configuration: entry=t=0, horizon=5 days
──────────────────────────────────────────────────────────────────────

Quintile Performance:
quintile  n_events  mean_return  std_return
       1       837     0.005246    0.082975
       2       836    -0.001420    0.074629
       3       836     0.006284    0.082711
       4       835     0.004174    0.077982
       5       836     0.002384    0.073456

📊 Long-Short Analysis:
  Q5 (Info) return:        0.0024 (  0.24%)
  Q1 (Liquidity) return:   0.0052 (  0.52%)
  Q5-Q1 spread:           -0.0029 ( -0.29%)

  Monotonic pattern: ✗ NO

──────────────────────────────────────────────────────────────────────
Configuration: entry=t=0, horizon=20 days
──────────────────────────────────────────────────────────────────────

Quintile Performance:
quintile  n_events  mean_return  std_return
       1       834     0.010174    0.117814
       2       824     0.013901    0.100312
       3       828     0.016665    0.098320
       4       830     0.018654    0.107133
       5       831     0.013479    0.098783

📊 Long-Short Analysis:
  Q5 (Info) return:        0.0135 (  1.35%)
  Q1 (Liquidity) return:   0.0102 (  1.02%)
  Q5-Q1 spread:            0.0033 (  0.33%)

  Monotonic pattern: ✗ NO

──────────────────────────────────────────────────────────────────────
Configuration: entry=t=+1, horizon=5 days
──────────────────────────────────────────────────────────────────────

Quintile Performance:
quintile  n_events  mean_return  std_return
       1       837     0.003630    0.065767
       2       836    -0.001540    0.061914
       3       835     0.006064    0.062657
       4       835     0.005389    0.061644
       5       836     0.001381    0.059326

📊 Long-Short Analysis:
  Q5 (Info) return:        0.0014 (  0.14%)
  Q1 (Liquidity) return:   0.0036 (  0.36%)
  Q5-Q1 spread:           -0.0022 ( -0.22%)

  Monotonic pattern: ✗ NO

──────────────────────────────────────────────────────────────────────
Configuration: entry=t=+1, horizon=20 days
──────────────────────────────────────────────────────────────────────

Quintile Performance:
quintile  n_events  mean_return  std_return
       1       834     0.009333    0.105379
       2       823     0.014619    0.093264
       3       828     0.015173    0.082302
       4       829     0.018763    0.092574
       5       830     0.012146    0.089378

📊 Long-Short Analysis:
  Q5 (Info) return:        0.0121 (  1.21%)
  Q1 (Liquidity) return:   0.0093 (  0.93%)
  Q5-Q1 spread:            0.0028 (  0.28%)

  Monotonic pattern: ✗ NO

======================================================================
EXTREME BINS ANALYSIS
======================================================================

Testing Top/Bottom 10%, 15%, 20% instead of quintiles
Rationale: Info trading concentrated in extreme events

──────────────────────────────────────────────────────────────────────
Top 10% vs Bottom 10%
──────────────────────────────────────────────────────────────────────
  Top 10%: 418 events (pressure_score >= 0.305)
  Bottom 10%: 418 events (pressure_score <= -0.060)

  t=0, 20d horizon:
    Top 10%:      0.0078 (  0.78%)
    Bottom 10%:   0.0094 (  0.94%)
    Spread:         -0.0016 ( -0.16%) ✗

  t=+1, 20d horizon:
    Top 10%:      0.0102 (  1.02%)
    Bottom 10%:   0.0056 (  0.56%)
    Spread:          0.0046 (  0.46%) ✗

──────────────────────────────────────────────────────────────────────
Top 15% vs Bottom 15%
──────────────────────────────────────────────────────────────────────
  Top 15%: 627 events (pressure_score >= 0.271)
  Bottom 15%: 627 events (pressure_score <= -0.026)

  t=0, 20d horizon:
    Top 15%:      0.0116 (  1.16%)
    Bottom 15%:   0.0069 (  0.69%)
    Spread:          0.0047 (  0.47%) ✗

  t=+1, 20d horizon:
    Top 15%:      0.0122 (  1.22%)
    Bottom 15%:   0.0048 (  0.48%)
    Spread:          0.0074 (  0.74%) ✓

──────────────────────────────────────────────────────────────────────
Top 20% vs Bottom 20%
──────────────────────────────────────────────────────────────────────
  Top 20%: 836 events (pressure_score >= 0.245)
  Bottom 20%: 836 events (pressure_score <= 0.002)

  t=0, 20d horizon:
    Top 20%:      0.0135 (  1.35%)
    Bottom 20%:   0.0103 (  1.03%)
    Spread:          0.0032 (  0.32%) ✗

  t=+1, 20d horizon:
    Top 20%:      0.0121 (  1.21%)
    Bottom 20%:   0.0094 (  0.94%)
    Spread:          0.0028 (  0.28%) ✗

✓ Saved results to: data/results/

======================================================================
INTERPRETATION GUIDE
======================================================================

Expected pattern if pressure_score works:
- Q5 (high info) should show DRIFT (positive returns)
- Q1 (high liquidity) should show REVERSION (negative/lower returns)
- Q5-Q1 should be significantly positive
- Pattern should be monotonic (Q1 < Q2 < Q3 < Q4 < Q5)

This is REAL DATA - results matter!
    

======================================================================
Analysis Complete!
======================================================================