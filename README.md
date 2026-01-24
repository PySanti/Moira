# Moira

El objetivo de este proyecto es crear un bot que se conectara con polymarket para apostar contra la temperatura maxima de una ciudad en un dia especifico.

# Desarrollo de V0

![Version 0 image](./images/v0.png)

## Desarrollo de funcion para consulta a API

### Definicion y refinamiento de features

Nota: es importante tener en cuenta las horas de ejecucion del bot, esto por que el bot se entrenara con data conseguida al final de los dias, entonces el bot mientras mas hacia el final del dia se ejecute, mas preciso sera por que mas se ajustara a su contexto de entrenamiento.

En esta seccion definire las features que se utilizaran para predecir la temperatura de un dia X + 1 a partir de data del dia X.

Empezare con una cantidad reducida de features para ampliar posiblemente en el futuro, mientras mas features, mas complicado construir la funcion.


| Nombre de feature                    |         Unidad | Rango de valores (típico)         | Significado                                                                                           |
| ------------------------------------ | -------------: | --------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Tmax_día_x**                       |             °C | ~ -50 a 55                        | Máxima observada el día *x*; principal señal de persistencia térmica.                                 |
| **Tmin_día_x**                       |             °C | ~ -60 a 35                        | Mínima del día *x*; refleja enfriamiento nocturno y masa de aire.                                     |
| **Tmedia_día_x**                     |             °C | ~ -55 a 45                        | Promedio térmico del día *x* (p.ej. (Tmax+Tmin)/2); estado térmico general.                           |
| **ΔTmax_1d = Tmax(x) − Tmax(x−1)**   |             °C | ~ -20 a 20                        | Tendencia/cambio reciente; captura entradas de aire, frentes y transiciones.                          |
| **MA_Tmax_3d (media móvil 3 días)**  |             °C | ~ -50 a 55                        | Inercia térmica de corto plazo; reduce ruido diario.                                                  |
| **DTR_x = Tmax(x) − Tmin(x)**        |             °C | ~ 0 a 25 (puede >30)              | Amplitud térmica diaria; proxy de nubosidad/humedad/mezcla atmosférica.                               |
| **HR_media_día_x**                   |              % | 0 a 100                           | Humedad relativa media; modula calentamiento diurno y nubosidad/convección.                           |
| **Punto_de_rocío_día_x (Td)**        |             °C | ~ -60 a 30 (puede >30 en trópico) | Contenido real de vapor de agua; suele ser más estable y predictivo que HR.                           |
| **Presión_media_día_x (SLP)**        |            hPa | ~ 870 a 1085                      | Estado sinótico (altas/bajas); asociado a estabilidad, nubosidad y frentes.                           |
| **ΔPresión_24h = SLP(x) − SLP(x−1)** |            hPa | ~ -20 a 20                        | Cambio sinótico rápido; caída/subida suele anticipar cambios de tiempo y temperatura.                 |
| **Viento_vel_media_día_x**           |            m/s | 0 a 30 (rachas mayores)           | Mezcla y advección; puede reducir o aumentar Tmax según origen del aire.                              |
| **Viento_dir_día_x_sin**             |              — | -1 a 1                            | Componente circular de dirección; permite al modelo “entender” la dirección sin saltos 359→0.         |
| **Viento_dir_día_x_cos**             |              — | -1 a 1                            | Segundo componente circular de dirección; junto con *sin* representa el ángulo completo.              |
| **Nubosidad_media_día_x**            | % (o fracción) | 0 a 100 (o 0 a 1)                 | Cobertura de nubes; controla radiación entrante y por tanto el calentamiento diurno.                  |
| **Precipitación_acum_día_x**         |         mm/día | 0 a 300+ (depende del clima)      | Lluvia/tormentas enfrían por evaporación y suelen venir con nubosidad; afecta Tmax del día siguiente. |

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

