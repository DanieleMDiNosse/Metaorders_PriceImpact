## Best Vs Second Fit Review

_Parameter intervals below are percentile bootstrap confidence intervals from the nonparametric full-sample bootstrap with alpha=0.0500 and 50 requested replicates. Each replicate reruns xmin selection and the models in [power_law, lognormal, truncated_power_law]. The machine-readable fit summary also stores bootstrap standard deviations for each reported parameter._

| Group | Metric | Best by AIC | Best fit parameters | 2nd | 2nd fit parameters | LR for best vs 2nd | p-value | Review |
|---|---|---|---|---|---|---:|---:|---|
| client | durations_minutes | truncated_power_law | alpha=0.9926 [0.8820, 1.0609]; lambda=0.0120 [0.0115, 0.0126] | lognormal | mu=4.0545 [4.0247, 4.1042]; sigma=0.6974 [0.6794, 0.7104] | 47.48 | 3.6e-11 | decisive |
| proprietary | durations_minutes | lognormal | mu=3.5921 [3.5197, 3.6445]; sigma=0.7805 [0.7626, 0.8017] | truncated_power_law | alpha=1.4733 [1.3988, 1.5355]; lambda=0.0111 [0.0107, 0.0118] | 4.78 | 0.5511 | tie / undecided |
| client | inter_arrivals_minutes | truncated_power_law | alpha=0.9774 [0.4979, 1.1152]; lambda=0.0580 [0.0530, 0.0733] | lognormal | mu=1.9404 [1.6513, 2.4468]; sigma=0.8911 [0.7117, 0.9929] | 2930.58 | ~0 | decisive |
| proprietary | inter_arrivals_minutes | truncated_power_law | alpha=1.2348 [1.1223, 1.2798]; lambda=0.0551 [0.0535, 0.0590] | lognormal | mu=1.7124 [1.5899, 1.9364]; sigma=0.9216 [0.8511, 0.9601] | 2431.79 | ~0 | decisive |
| client | meta_volumes | truncated_power_law | alpha=1.8760 [1.8576, 1.8922]; lambda=2.0e-07 [1.7e-07, 2.3e-07] | lognormal | mu=5.4883 [3.7261, 6.4449]; sigma=2.6743 [2.4994, 2.9842] | 47.53 | 3.4e-07 | decisive |
| proprietary | meta_volumes | truncated_power_law | alpha=1.8598 [1.8510, 1.8704]; lambda=3.2e-07 [3.0e-07, 3.5e-07] | lognormal | mu=3.2490 [2.6348, 4.1370]; sigma=3.0073 [2.8569, 3.1124] | 305.10 | 1.6e-124 | decisive |
| client | q_over_v | truncated_power_law | alpha=1.9200 [1.7263, 2.0228]; lambda=23.9470 [21.2471, 28.5522] | lognormal | mu=-4.9722 [-5.7129, -4.9146]; sigma=0.9324 [0.9128, 1.1325] | 5.05 | 0.2511 | tie / undecided |
| proprietary | q_over_v | lognormal | mu=-5.7773 [-5.8864, -5.7071]; sigma=0.9545 [0.9320, 0.9832] | truncated_power_law | alpha=2.0910 [2.0182, 2.1444]; lambda=41.0217 [37.9803, 44.6746] | 80.07 | 2.5e-12 | decisive |
| client | participation_rates | lognormal | mu=-2.1999 [-2.2217, -2.1764]; sigma=0.6815 [0.6715, 0.6901] | truncated_power_law | alpha=0.9526 [0.8897, 1.0100]; lambda=6.6254 [6.3975, 6.8641] | 26.24 | 0.0082 | decisive |
| proprietary | participation_rates | truncated_power_law | alpha=0.3102 [0.2731, 0.3537]; lambda=6.4970 [6.3991, 6.6041] | exponential | lambda=7.3023 | 67.08 | 2.5e-10 | decisive |

## Distribution Means And Medians

| Group | Metric | Sample size | Mean | Median |
|---|---|---:|---:|---:|
| client | durations_minutes | 256371 | 36.8402 | 19.0411 |
| proprietary | durations_minutes | 588543 | 19.1341 | 8.8617 |
| client | inter_arrivals_minutes | 3449403 | 2.7381 | 0.5665 |
| proprietary | inter_arrivals_minutes | 5297419 | 2.1258 | 0.4085 |
| client | meta_volumes | 256371 | 38933.1611 | 7172.0000 |
| proprietary | meta_volumes | 588543 | 30800.1527 | 6519.0000 |
| client | q_over_v | 256371 | 0.0042 | 0.0018 |
| proprietary | q_over_v | 588543 | 0.0034 | 0.0018 |
| client | participation_rates | 256371 | 0.0844 | 0.0499 |
| proprietary | participation_rates | 588543 | 0.1362 | 0.0949 |
