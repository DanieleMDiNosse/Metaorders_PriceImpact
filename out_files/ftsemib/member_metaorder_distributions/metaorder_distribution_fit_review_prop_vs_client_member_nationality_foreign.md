## Best Vs Second Fit Review

_Parameter intervals below are percentile bootstrap confidence intervals from the nonparametric full-sample bootstrap with alpha=0.0500 and 50 requested replicates. Each replicate reruns xmin selection and the models in [power_law, lognormal, truncated_power_law]. The machine-readable fit summary also stores bootstrap standard deviations for each reported parameter._

| Group | Metric | Best by AIC | Best fit parameters | 2nd | 2nd fit parameters | LR for best vs 2nd | p-value | Review |
|---|---|---|---|---|---|---:|---:|---|
| client | durations_minutes | truncated_power_law | alpha=0.9857 [0.9104, 1.0618]; lambda=0.0116 [0.0111, 0.0121] | lognormal | mu=4.0832 [4.0422, 4.1240]; sigma=0.7007 [0.6879, 0.7153] | 61.10 | 7.1e-22 | decisive |
| proprietary | durations_minutes | lognormal | mu=3.5744 [3.5107, 3.6263]; sigma=0.7859 [0.7705, 0.8058] | truncated_power_law | alpha=1.4958 [1.4266, 1.5410]; lambda=0.0110 [0.0105, 0.0116] | 2.77 | 0.7245 | tie / undecided |
| client | inter_arrivals_minutes | truncated_power_law | alpha=1.0179 [0.6755, 1.1122]; lambda=0.0576 [0.0539, 0.0688] | lognormal | mu=1.9036 [1.7203, 2.3269]; sigma=0.8967 [0.7493, 0.9614] | 2507.06 | ~0 | decisive |
| proprietary | inter_arrivals_minutes | truncated_power_law | alpha=1.2361 [1.1227, 1.2798]; lambda=0.0556 [0.0540, 0.0597] | lognormal | mu=1.7167 [1.5987, 1.9437]; sigma=0.9167 [0.8441, 0.9531] | 2331.35 | ~0 | decisive |
| client | meta_volumes | truncated_power_law | alpha=1.9013 [1.8778, 1.9217]; lambda=1.7e-07 [1.4e-07, 2.1e-07] | lognormal | mu=4.6533 [3.3488, 5.5698]; sigma=2.8152 [2.6480, 3.0275] | 41.54 | 7.7e-08 | decisive |
| proprietary | meta_volumes | truncated_power_law | alpha=1.8592 [1.8493, 1.8663]; lambda=3.2e-07 [3.1e-07, 3.4e-07] | lognormal | mu=3.2873 [2.7507, 3.7606]; sigma=3.0008 [2.9252, 3.0870] | 303.99 | 1.0e-122 | decisive |
| client | q_over_v | truncated_power_law | alpha=1.8928 [1.7046, 2.0100]; lambda=25.4691 [21.9125, 29.9697] | lognormal | mu=-4.9340 [-5.6107, -4.8191]; sigma=0.9137 [0.8838, 1.0949] | 5.25 | 0.2408 | tie / undecided |
| proprietary | q_over_v | lognormal | mu=-5.7730 [-5.8842, -5.6838]; sigma=0.9518 [0.9254, 0.9807] | truncated_power_law | alpha=2.0859 [1.9676, 2.1452]; lambda=41.5295 [38.3056, 46.6622] | 79.39 | 5.0e-12 | decisive |
| client | participation_rates | lognormal | mu=-2.2012 [-2.2159, -2.1782]; sigma=0.6722 [0.6627, 0.6780] | truncated_power_law | alpha=0.9054 [0.8344, 0.9439]; lambda=6.9667 [6.8077, 7.2384] | 39.74 | 8.5e-05 | decisive |
| proprietary | participation_rates | truncated_power_law | alpha=0.3093 [0.2674, 0.3541]; lambda=6.5008 [6.3707, 6.6149] | exponential | lambda=7.3034 | 66.33 | 3.2e-10 | decisive |

## Distribution Means And Medians

| Group | Metric | Sample size | Mean | Median |
|---|---|---:|---:|---:|
| client | durations_minutes | 232260 | 36.5707 | 18.4401 |
| proprietary | durations_minutes | 586450 | 19.0141 | 8.8280 |
| client | inter_arrivals_minutes | 3237657 | 2.6235 | 0.5399 |
| proprietary | inter_arrivals_minutes | 5274743 | 2.1140 | 0.4085 |
| client | meta_volumes | 232260 | 38774.0964 | 7060.0000 |
| proprietary | meta_volumes | 586450 | 30734.4106 | 6509.0000 |
| client | q_over_v | 232260 | 0.0043 | 0.0019 |
| proprietary | q_over_v | 586450 | 0.0034 | 0.0018 |
| client | participation_rates | 232260 | 0.0865 | 0.0536 |
| proprietary | participation_rates | 586450 | 0.1364 | 0.0952 |
