# Moira

El objetivo de este proyecto es crear un bot que se conectara con polymarket para apostar contra la temperatura maxima de una ciudad en un dia especifico.

# Desarrollo de V0

![Version 0 image](./images/v0.png)

## Desarrollo de funcion para consulta a API

### Definicion y refinamiento de features

En esta seccion definire las features que se utilizaran para predecir la temperatura de un dia X + 1 a partir de data del dia X.



| Grupo                         | Feature (columna sugerida) |   Ventana / timestamp |  Unidades | Nota (por qué sirve)                           |
| ----------------------------- | -------------------------- | --------------------: | --------: | ---------------------------------------------- |
| **Objetivo**                  | **y = Tmax_Xplus1**        |         Día X+1 (24h) |        °F | Variable a predecir.                           |
| **Temperatura (obs)**         | T_00                       |               X 00:00 |        °F | Estado térmico nocturno.                       |
|                               | T_06                       |               X 06:00 |        °F | Cercano a mínima/amanecer.                     |
|                               | **T_12**                   |               X 12:00 |        °F | Estado térmico al momento de predicción.       |
|                               | Tmin_X_sofar               |            min(00–12) |        °F | Mínima “observada hasta 12”.                   |
|                               | T_mean_0_12                |           mean(00–12) |        °F | Promedio parcial, proxy de inercia.            |
|                               | T_std_0_12                 |            std(00–12) |        °F | Variabilidad intramañana.                      |
| **Humedad (obs)**             | Td_00 (dewpoint_00)        |               X 00:00 |        °F | Contenido de humedad nocturna.                 |
|                               | Td_06                      |               X 06:00 |        °F | Humedad al amanecer.                           |
|                               | **Td_12**                  |               X 12:00 |        °F | Humedad al mediodía (impacta nubes/evap).      |
|                               | RH_12                      |               X 12:00 |         % | Si existe, úsala junto con Td.                 |
|                               | q_12                       |               X 12:00 |      g/kg | Alternativa robusta a RH.                      |
|                               | **VPD_12**                 |               X 12:00 |      inHg | Proxy directo de sequedad (controla Tmax).     |
|                               | RH_mean_0_12               |           mean(00–12) |         % | Versión parcial (no diaria completa).          |
| **Nubes / radiación**         | cloud_total_12             |               X 12:00 |         % | Controla calentamiento diurno.                 |
|                               | cloud_low_12               |               X 12:00 |         % | Nube baja reduce Tmax fuerte.                  |
|                               | cloud_mean_0_12            |           mean(00–12) |         % | Persistencia de nubosidad.                     |
|                               | GHI_sum_0_12               |            sum(00–12) | Btu/ft²·h | Energía entrante acumulada (parcial).          |
|                               | DNI_sum_0_12               |            sum(00–12) | Btu/ft²·h | Radiación directa (cielos despejados).         |
|                               | DHI_sum_0_12               |            sum(00–12) | Btu/ft²·h | Difusa (cielos turbios/nublados).              |
|                               | Rn_sum_0_12                |            sum(00–12) | Btu/ft²·h | Balance neto parcial.                          |
|                               | sunshine_h_0_12            |      acumulado(00–12) |         h | Insolación parcial (no día completo).          |
| **Viento (dinámica)**         | U_12                       |               X 12:00 |       m/s | Mejor que “dirección” cruda.                   |
|                               | V_12                       |               X 12:00 |       m/s | Idem.                                          |
|                               | wind_speed_mean_0_12       |           mean(00–12) |       m/s | Mezcla/advección.                              |
|                               | gust_max_0_12              |            max(00–12) |       m/s | Señal de mezcla fuerte/frentes.                |
| **Presión**                   | SLP_12                     |               X 12:00 |      inHg | Régimen sinóptico.                             |
|                               | P_sfc_12                   |               X 12:00 |      inHg | Alternativa si no hay SLP.                     |
|                               | dP_3h                      |             (12 − 09) |      inHg | Tendencia corta (paso de sistemas).            |
|                               | dP_6h                      |             (12 − 06) |      inHg | Tendencia mañana–mediodía.                     |
|                               | dP_24h                     |        (12 − 12 ayer) |      inHg | Cambio sinóptico día a día.                    |
| **Precip / visibilidad**      | precip_sum_0_12            |            sum(00–12) |        in | Lluvia enfría / nubosidad persistente.         |
|                               | precip_intensity_max_0_12  |            max(00–12) |      in/h | Convección/tormenta (cambio de régimen).       |
|                               | visibility_12              |               X 12:00 |        mi | Proxy de niebla/aerosoles/humedad.             |
|                               | fog_flag_0_12              |            any(00–12) |       0/1 | Nube baja/estratos matinales.                  |
|                               | storm_flag_0_12            |            any(00–12) |       0/1 | Convección: nubosidad + outflows.              |
| **Suelo / criosfera**         | soil_temp_12               |               X 12:00 |        °F | Memoria térmica superficie.                    |
|                               | soil_moist_12              |               X 12:00 |     m³/m³ | Controla partición sensible/latente.           |
|                               | snow_depth                 |               X 12:00 |        in | Alta albedo + enfriamiento.                    |
|                               | snow_cover                 |               X 12:00 |         % | Idem.                                          |
| **Aerosoles (opc.)**          | AOD_12                     |               X 12:00 |         — | Modula radiación; depende calidad del dato.    |
|                               | PM25_12                    |               X 12:00 |     µg/m³ | Puede correlacionar con estabilidad/radiación. |
|                               | O3_12                      |               X 12:00 |       ppb | Útil en algunos climas, opcional.              |
| **Derivadas (tendencias)**    | dT_06_12                   |         (T_12 − T_06) |        °F | “Calentamiento matutino” → clave para Tmax.    |
|                               | dTd_06_12                  |       (Td_12 − Td_06) |        °F | Cambio de masa de aire/humedad.                |
|                               | dU_06_12                   |         (U_12 − U_06) |       m/s | Señal de giro/refuerzo del flujo.              |
|                               | dV_06_12                   |         (V_12 − V_06) |       m/s | Idem.                                          |
|                               | partial_range              | (T_12 − Tmin_X_sofar) |        °F | Amplitud parcial (no uses Tmax−Tmin).          |
| **Lags (memoria)**            | Tmax_Xm1                   |               día X−1 |        °F | Persistencia diaria.                           |
|                               | Tmax_Xm2                   |               día X−2 |        °F | Persistencia extendida.                        |
|                               | T12_Xm1                    |             X−1 12:00 |        °F | Estado al mismo corte horario.                 |
|                               | Td12_Xm1                   |             X−1 12:00 |        °F | Humedad persistente.                           |
|                               | SLP12_Xm1                  |             X−1 12:00 |      inHg | Régimen sinóptico persistente.                 |
|                               | roll_Tmax_3                |        últimos 3 días |        °F | Suaviza ruido, capta régimen.                  |
|                               | roll_Tmax_7                |        últimos 7 días |        °F | Estacionalidad local de corto plazo.           |
|                               | precip_3d_to_Xm1           | acum 3 días hasta X−1 |        in | Humedad del suelo / nubosidad previa.          |
|                               | precip_7d_to_Xm1           | acum 7 días hasta X−1 |        in | Memoria hidrológica.                           |
| **Tiempo/año (multi-ciudad)** | DoY                        |                     X |     1–365 | Estación.                                      |
|                               | sin_DoY                    |                     X |         — | Estación (cíclico).                            |
|                               | cos_DoY                    |                     X |         — | Estación (cíclico).                            |
|                               | daylength                  |                     X |         h | Control radiativo astronómico.                 |
| **Geografía (multi-ciudad)**  | lat                        |                  fijo |         ° | Gradiente climático.                           |
|                               | lon                        |                  fijo |         ° | Régimen regional.                              |
|                               | **altitude**               |                  fijo |        ft | Ajusta lapse rate / clima local.               |
|                               | dist_to_sea                |                  fijo |        mi | Moderación marítima.                           |
|                               | NDVI                       |    mensual/estacional |         — | Cobertura vegetal (albedo/ET).                 |
|                               | urban_proxy                |                  fijo |       0–1 | Isla de calor urbana (muy útil multi-ciudad).  |
| **Upper-air (recomendado)**   | T850_12Z                   |         cercano a 12Z |        °F | Advección de masa de aire (top predictor).     |
|                               | q850_12Z                   |                   12Z |      g/kg | Humedad en niveles bajos.                      |
|                               | U850_12Z                   |                   12Z |       m/s | Flujo sinóptico bajo.                          |
|                               | V850_12Z                   |                   12Z |       m/s | Idem.                                          |
|                               | Z500_12Z                   |                   12Z |        ft | Patrón de ondas/altas-bajas.                   |
|                               | thickness_1000_500_12Z     |                   12Z |        ft | Proxy de temperatura media columna.            |
|                               | dT850_24h                  |     (T850 hoy − ayer) |        °F | Tendencia sinóptica (muy fuerte).              |
|                               | dZ500_24h                  |     (Z500 hoy − ayer) |        ft | Cambio de patrón.                              |


### Creacion de funcion para consulta a api

### Testeo de funcion para

La funcion debe ser testeada para:

* Posibles bloqueos por rate limiting 
* Alcance de fechas
* Null values

## Creacion de pipeline de preprocesamiento

## Entrenamiento de modelo

* Definir algoritmo de ML
* Obtener data de entrenamiento
* Seleccion de hiperparametros + entrenamiento

# Desarrollo de V1

![Version 1 image](./images/v1.png)

# Desarrollo de V2

![Version 2 image](./images/v2.png)

# Desarrollo de V3

![Version 3 image](./images/v3.png)

