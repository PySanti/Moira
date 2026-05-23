# Sprint 4 - V0: Estudio de utilidad predictiva de features
Se analizo `data/processed/sprint3.csv` con foco en explicar por que el rendimiento fuera de muestra no siempre cumple expectativas.

Metodologia:
- Split del estudio: train `1980-2020`, evaluacion externa `2021-2025`, y nota separada para 2026.
- Ranking por `utility_score` compuesto (correlaciones, mutual information, importancia por permutacion, estabilidad temporal y penalizaciones por missingness, redundancia y drift).
- Criterio principal: utilidad predictiva practica sobre modelo, no solo correlacion lineal.

Top 10 features mas utiles:

| Rank | Feature | Utility score | Abs Pearson | Abs Spearman | Mutual Info | Drift KS |
| ---: | ------- | ------------: | ----------: | -----------: | ----------: | -------: |
| 1 | `Temp_23h_x` | `0.8288` | `0.9446` | `0.9441` | `1.1139` | `0.0456` |
| 2 | `Temp_min_last_6h` | `0.7443` | `0.9433` | `0.9430` | `1.1010` | `0.0455` |
| 3 | `Temp_mean_last_6h` | `0.6604` | `0.9388` | `0.9387` | `1.0447` | `0.0462` |
| 4 | `Temp_21h_x` | `0.6543` | `0.9370` | `0.9368` | `1.0169` | `0.0443` |
| 5 | `Temp_max_last_6h` | `0.6392` | `0.9309` | `0.9311` | `0.9713` | `0.0470` |
| 6 | `Temp_18h_x` | `0.6312` | `0.9276` | `0.9279` | `0.9390` | `0.0452` |
| 7 | `Tmean_so_far_23h_x` | `0.6189` | `0.9191` | `0.9217` | `0.9064` | `0.0451` |
| 8 | `Tmax_so_far_23h_x` | `0.6101` | `0.9136` | `0.9159` | `0.8782` | `0.0426` |
| 9 | `Tmin_so_far_23h_x` | `0.6055` | `0.9101` | `0.9129` | `0.8764` | `0.0478` |
| 10 | `Temp_23h_ma3` | `0.6042` | `0.9146` | `0.9164` | `0.8627` | `0.0502` |

Top 10 features menos utiles:

| Rank | Feature | Utility score | Missing rate | Redundancy max corr | Drift KS |
| ---: | ------- | ------------: | -----------: | ------------------: | -------: |
| 1 | `Cloud_mean_last_6h` | `-0.0183` | `0.0009` | `0.8835` | `0.1324` |
| 2 | `extreme_heat_flag` | `-0.0193` | `0.0489` | `0.6317` | `0.0020` |
| 3 | `extreme_cold_flag` | `-0.0231` | `0.0489` | `0.5474` | `0.0352` |
| 4 | `HR_min_00_23h_x` | `-0.0233` | `0.0000` | `0.9011` | `0.1336` |
| 5 | `Temp_dewpoint_spread_23h` | `-0.0240` | `0.0003` | `0.9792` | `0.1147` |
| 6 | `Temp_dewpoint_spread_mean_00_23h` | `-0.0500` | `0.0000` | `0.9750` | `0.1422` |
| 7 | `tmax_anomaly_x` | `-0.0600` | `0.0489` | `0.9991` | `0.0549` |
| 8 | `tmax_anomaly_vs_doy_plus1` | `-0.0620` | `0.0489` | `0.9991` | `0.0538` |
| 9 | `climatology_tmax_delta_doy_plus1_minus_x` | `-0.0680` | `0.0489` | `0.7355` | `0.0636` |
| 10 | `Cloud_23h_x` | `-0.1967` | `0.0524` | `0.8835` | `0.1557` |

Lecturas clave:
- La familia con mayor utilidad promedio fue `astronomical`.
- La familia con menor utilidad promedio fue `other`.
- Las features con baja utilidad suelen combinar baja senal, alta redundancia o drift temporal significativo.
- El deterioro fuera de muestra se asocia a cambio de distribucion y menor estabilidad temporal en parte del set de variables.

Graficos:

![Top 10 utility score](./experiments/v0-sprint4/correlation_1.0/plots/top_10_utility_score.png)
![Bottom 10 utility score](./experiments/v0-sprint4/correlation_1.0/plots/bottom_10_utility_score.png)
![Pearson vs Spearman](./experiments/v0-sprint4/correlation_1.0/plots/pearson_vs_spearman.png)
![Mutual information top 30](./experiments/v0-sprint4/correlation_1.0/plots/mutual_information_top_30.png)
![Permutation importance top 30](./experiments/v0-sprint4/correlation_1.0/plots/permutation_importance_top_30.png)
![Missingness by feature](./experiments/v0-sprint4/correlation_1.0/plots/missingness_by_feature.png)
![Top features heatmap](./experiments/v0-sprint4/correlation_1.0/plots/top_features_correlation_heatmap.png)
![Utility por familia](./experiments/v0-sprint4/correlation_1.0/plots/family_utility_summary.png)
