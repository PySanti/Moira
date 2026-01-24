# Moira

El objetivo de este proyecto es crear un bot que se conectara con polymarket para apostar contra la temperatura maxima de una ciudad en un dia especifico.

# Desarrollo de V0

![Version 0 image](./images/v0.png)

## Desarrollo de funcion para consulta a API

### Definicion y refinamiento de features

Nota: es importante tener en cuenta las horas de ejecucion del bot, esto por que el bot se entrenara con data conseguida al final de los dias, entonces el bot mientras mas hacia el final del dia se ejecute, mas preciso sera por que mas se ajustara a su contexto de entrenamiento.

En esta seccion definire las features que se utilizaran para predecir la temperatura de un dia X + 1 a partir de data del dia X.

Empezare con una cantidad reducida de features para ampliar posiblemente en el futuro, mientras mas features, mas complicado construir la funcion.


| Nombre de feature                  |         Unidad | Rango de valores (típico) | Significado (incluye cálculo)                                                                              |
| ---------------------------------- | -------------: | ------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Tmax_día_x**                     |             °C | ~ -50 a 55                | **Cálculo:** `Tmax[x]`. Máxima del día *x* (persistencia térmica).                                         |
| **Tmin_día_x**                     |             °C | ~ -60 a 35                | **Cálculo:** `Tmin[x]`. Mínima del día *x* (masa de aire/enfriamiento nocturno).                           |
| **Tmedia_día_x**                   |             °C | ~ -55 a 45                | **Cálculo:** `(Tmax[x] + Tmin[x]) / 2` (o `Tmed[x]`). Estado térmico general.                              |
| **ΔTmax_1d**                       |             °C | ~ -20 a 20                | **Cálculo:** `Tmax[x] − Tmax[x−1]`. Tendencia/cambio reciente.                                             |
| **MA_Tmax_3d**                     |             °C | ~ -50 a 55                | **Cálculo:** `(Tmax[x] + Tmax[x−1] + Tmax[x−2]) / 3`. Inercia térmica de corto plazo.                      |
| **DTR_x**                          |             °C | ~ 0 a 25 (puede >30)      | **Cálculo:** `Tmax[x] − Tmin[x]`. Amplitud térmica; proxy nubosidad/humedad.                               |
| **HR_media_día_x**                 |              % | 0 a 100                   | **Cálculo:** `HR_mean[x]`. Humedad relativa media diaria.                                                  |
| **Punto_de_rocío_día_x (Td)**      |             °C | ~ -60 a 30+               | **Cálculo:** `Td[x]` (preferible si viene en el dataset). Contenido real de vapor de agua.                 |
| **Presión_media_día_x (SLP)**      |            hPa | ~ 870 a 1085              | **Cálculo:** `SLP_mean[x]`. Señal sinótica (altas/bajas).                                                  |
| **ΔPresión_24h**                   |            hPa | ~ -20 a 20                | **Cálculo:** `SLP_mean[x] − SLP_mean[x−1]`. Cambio sinótico rápido.                                        |
| **Viento_vel_media_día_x**         |            m/s | 0 a 30 (rachas mayores)   | **Cálculo:** `wind_speed_mean[x]`. Mezcla/advección.                                                       |
| **Viento_dir_sin(x)**              |              — | -1 a 1                    | **Cálculo:** `sin(2π * wind_dir_deg[x] / 360)`. Codificación circular de dirección.                        |
| **Viento_dir_cos(x)**              |              — | -1 a 1                    | **Cálculo:** `cos(2π * wind_dir_deg[x] / 360)`. Codificación circular de dirección.                        |
| **Nubosidad_media_día_x**          | % (o fracción) | 0–100 (o 0–1)             | **Cálculo:** `cloud_cover_mean[x]`. Control de radiación entrante.                                         |
| **Precipitación_acum_día_x**       |         mm/día | 0 a 300+                  | **Cálculo:** `precip_sum[x]`. Efecto de lluvia/nubosidad/evaporación.                                      |
| **t_max_x+1 (si está disponible)** |             °C | ~ -50 a 55                | **Cálculo:** `Tmax[x+1]`. **Label/objetivo** para entrenamiento; **no usar como feature** en inferencia.   |
| **día (del mes)**                  |           1–31 | 1 a 31                    | **Cálculo:** `day_of_month(fecha)`. Calendario (efecto débil; útil como índice).                           |
| **mes**                            |           1–12 | 1 a 12                    | **Cálculo:** `month(fecha)`. Estacionalidad mensual (mejor usar DOY cíclico abajo).                        |
| **año**                            |           YYYY | p.ej. 1950–2100           | **Cálculo:** `year(fecha)`. Tendencia de largo plazo/cambios en medición.                                  |
| **ciudad**                         |      categoría | N categorías              | **Cálculo:** ID/nombre. Se codifica (one-hot/target encoding/embeddings) para capturar climatología local. |
| **doy (día del año)**              |      1–365/366 | 1 a 366                   | **Cálculo:** `doy = día_del_año(fecha)`. Índice estacional más fino que “mes”.                             |
| **doy_sin**                        |              — | -1 a 1                    | **Cálculo:** `sin(2π * doy / 365)`. Estacionalidad en forma cíclica (diciembre cerca de enero).            |
| **doy_cos**                        |              — | -1 a 1                    | **Cálculo:** `cos(2π * doy / 365)`. Complementa `doy_sin` para representar el ciclo anual.                 |

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

