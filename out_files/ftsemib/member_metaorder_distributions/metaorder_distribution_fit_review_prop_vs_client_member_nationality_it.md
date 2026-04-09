## Best Vs Second Fit Review

_Parameter intervals below are percentile bootstrap confidence intervals from the nonparametric full-sample bootstrap with alpha=0.0500 and 50 requested replicates. Each replicate reruns xmin selection and the models in [power_law, lognormal, truncated_power_law]. The machine-readable fit summary also stores bootstrap standard deviations for each reported parameter._

| Group | Metric | Best by AIC | Best fit parameters | 2nd | 2nd fit parameters | LR for best vs 2nd | p-value | Review |
|---|---|---|---|---|---|---:|---:|---|
| client | durations_minutes | lognormal | mu=4.1118 [4.0588, 4.1609]; sigma=0.5563 [0.5299, 0.5782] | exponential | lambda=0.0244 | 8.16 | 0.1024 | tie / undecided |
| proprietary | durations_minutes | exponential | lambda=0.0223 | truncated_power_law | alpha=0.0728 [9.0e-07, 0.4783]; lambda=0.0216 [0.0172, 0.0232] | -0.04 | 0.9043 | tie / undecided |
| client | inter_arrivals_minutes | exponential | lambda=0.0828 | truncated_power_law | alpha=0.0326 [5.0e-08, 0.6520]; lambda=0.0817 [0.0624, 0.0826] | -0.21 | 0.6945 | tie / undecided |
| proprietary | inter_arrivals_minutes | truncated_power_law | alpha=1.0523 [0.2338, 1.0605]; lambda=0.0230 [0.0224, 0.0569] | lognormal | mu=-1.4108 [-1.4983, 2.3432]; sigma=2.9294 [0.9833, 2.9671] | 1080.45 | ~0 | decisive |
| client | meta_volumes | truncated_power_law | alpha=1.7247 [1.6927, 1.7553]; lambda=4.5e-07 [3.6e-07, 5.5e-07] | lognormal | mu=4.9578 [3.9246, 6.4240]; sigma=2.8283 [2.5281, 3.0762] | 30.69 | 2.8e-12 | decisive |
| proprietary | meta_volumes | lognormal | mu=6.6251 [4.1723, 8.2340]; sigma=2.3991 [1.9330, 2.9362] | truncated_power_law | alpha=1.7445 [1.6498, 1.8110]; lambda=3.6e-07 [2.3e-07, 7.7e-07] | 4.87 | 0.0733 | decisive |
| client | q_over_v | truncated_power_law | alpha=1.9495 [1.7883, 2.0161]; lambda=13.2809 [10.7515, 18.0744] | lognormal | mu=-7.3642 [-8.1698, -6.4610]; sigma=1.6207 [1.4218, 1.8038] | 7.49 | 0.0001 | decisive |
| proprietary | q_over_v | truncated_power_law | alpha=1.0994 [1.0224, 1.1473]; lambda=41.8459 [38.3647, 51.3114] | lognormal | mu=-6.7816 [-7.0307, -6.3974]; sigma=1.7332 [1.5152, 1.8325] | 30.99 | 1.2e-07 | decisive |
| client | participation_rates | truncated_power_law | alpha=0.7806 [0.5345, 0.9825]; lambda=5.2713 [4.7503, 5.8896] | exponential | lambda=7.4227 | 11.95 | 0.0067 | decisive |
| proprietary | participation_rates | truncated_power_law | alpha=0.9217 [0.7450, 0.9735]; lambda=3.5601 [3.0715, 4.7278] | lognormal | mu=-4.0449 [-4.2984, -2.7312]; sigma=1.8335 [1.1200, 1.9896] | 62.45 | 3.2e-23 | decisive |

## Distribution Means And Medians

| Group | Metric | Sample size | Mean | Median |
|---|---|---:|---:|---:|
| client | durations_minutes | 24111 | 39.4362 | 25.1627 |
| proprietary | durations_minutes | 2093 | 52.7444 | 42.4418 |
| client | inter_arrivals_minutes | 211746 | 4.4905 | 1.1535 |
| proprietary | inter_arrivals_minutes | 22676 | 4.8683 | 0.3935 |
| client | meta_volumes | 24111 | 40465.4223 | 8131.0000 |
| proprietary | meta_volumes | 2093 | 49220.7989 | 10950.0000 |
| client | q_over_v | 24111 | 0.0030 | 0.0011 |
| proprietary | q_over_v | 2093 | 0.0048 | 0.0014 |
| client | participation_rates | 24111 | 0.0635 | 0.0191 |
| proprietary | participation_rates | 2093 | 0.0679 | 0.0185 |
