# Forecasting de rendimiento de maiz

Proyecto de ciencia de datos para explorar y pronosticar el rendimiento de maiz
en Argentina a nivel administrativo (`adm_id`) y por campana agricola.

## Datos disponibles

Los CSV originales se conservan localmente en `AR/maize/` y no se versionan en
GitHub por su tamano y por las condiciones de distribucion de la fuente. Descarga
CY-Bench v1.10 desde
[Zenodo](https://zenodo.org/records/17279151) y coloca los CSV de maize para
Argentina en `AR/maize/`. La configuracion apunta a esa carpeta, por lo que no
hace falta mover ni duplicar los archivos.

- `yield_maize_AR.csv`: rendimiento, produccion, area y fechas de siembra/cosecha
  por `adm_id` y campana. Es el archivo objetivo inicial.
- `meteo_maize_AR.csv`: observaciones diarias de temperatura, precipitacion,
  radiacion, evapotranspiracion, VPD y balance hidrico.
- `soil_moisture_maize_AR.csv`: humedad del suelo diaria.
- `ndvi_maize_AR.csv`, `fpar_maize_AR.csv`: indices de vegetacion por fecha.
- `soil_maize_AR.csv`, `crop_calendar_maize_AR.csv`, `location_maize_AR.csv` y
  `crop_mask_maize_AR.csv`: atributos de suelo, calendario, ubicacion y area.

Los archivos diarios grandes tienen millones de filas. Se conservan por fuente y
grano. El rendimiento cubre 1970-2023, meteorologia 2001-2023 y humedad de suelo
2003-2023. Campanas y fechas de calendario permanecen en ejes separados; las
vistas estacionales solo asignan fechas a ventanas de campana y no agregan
features.

El proyecto usa CY-Bench v1.10 solo para maize en Argentina (`AR/maize`). Los
originales se preservan sin modificar. La preparacion escribe
copias Parquet en `data/interim/standardized/`, genera reportes en
`reports/data_quality/` y materializa el modelo estrella analitico en
`data/processed/star_schema/maize_star_schema.duckdb`. Los hechos conservan el
grano de cada fuente. La dimensión administrativa común se basa en las
correcciones de `adm_id` descritas para CY-Bench v1.1/1.2; conserva la unión de
IDs y deja vacíos los atributos de ubicación cuando no existen en el CSV de
location. Las campañas de rendimiento y las fechas ambientales permanecen en
dimensiones temporales separadas; el modelo estrella no es aún un dataset de
entrenamiento. `bridge_administrative_season` aproxima el cutoff local como el
punto medio de `sos/eos` estaticos; las vistas `vw_*_season_stage` agregan
`season_day`, `season_progress` y fases genericas `early/mid/late` a cada
observacion fechada. Hay 228 filas de rendimiento sin calendario y quedan sin
cutoff. La fecha de observacion se usa como proxy de disponibilidad.
La etapa de feature engineering compara cutoffs 25/50/75 con splits temporales;
las tablas candidatas y sus metricas quedan en DuckDB y
`reports/feature_engineering/`. El training mart final se crea solo despues de
revisar variables, unidades y rendimiento en validacion. El periodo 2019-2022 ya
se habia consultado en una corrida candidata inicial, por lo que las metricas
actuales de ese holdout son exploratorias, no un test final independiente. Las
hipotesis y unidades de features candidatas estan en
`metadata/feature_registry.json`. La tabla `model_input_candidates_cutoff_50`
se preparo antes de modeling con columnas nuevas de AWC, coordenadas y drenaje
aun pendientes de evaluacion; esa etapa de preparacion no entreno el modelo.

La etapa de modeling compara regresores con validacion rolling-origin y guarda
el pipeline experimental en `models/`; `reports/modeling/` documenta metricas y
limitaciones. El holdout actual es exploratorio, no una prueba final ciega.
El diccionario versionado de variables esta en `metadata/data_dictionary.json`;
la interpretacion de `mm0` como escenario sigue marcada como asumida y las
unidades no confirmadas permanecen explicitamente pendientes.

## Estructura

```text
AR/maize/                 CSV originales
data/interim/             copias estandarizadas en Parquet
data/processed/           salidas analiticas y futuros datos de modelado
data/processed/star_schema/ modelo estrella DuckDB
notebooks/                Data Understanding, preparacion, EDA y experimentos
src/maize_forecasting/    codigo reutilizable del proyecto
tests/                    pruebas
models/                   modelos serializados
reports/figures/          graficos exportados
reports/data_quality/     auditorias por fuente e informe de integracion
metadata/                 diccionario de variables y escenario de produccion
```

## Entorno local (Windows / PowerShell)

Se usa `venv`, incluido con Python; Conda no es necesario.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m jupyter lab
```

Si PowerShell bloquea la activacion, se puede invocar directamente el Python del
entorno: `.\.venv\Scripts\python.exe`.

Ejecutar las pruebas con:

```powershell
python -m pytest
```
