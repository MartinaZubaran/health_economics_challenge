# ============================================================================
# TRAINING STRATEGY PARA HEALTH ECONOMICS
# Universidad del Oeste (UNO) - Aplicaciones en Ciencia de Datos
# ============================================================================
# Este script implementa la estrategia de entrenamiento (Training Strategy)
# para el desafío de Health Economics.
#
# ¿QUÉ HACE ESTE SCRIPT?
# Toma el dataset con Feature Engineering y lo divide en diferentes conjuntos:
#
#   1. PRESENT: Datos del año 2021 (SIN variable objetivo) para hacer
#               predicciones de 2022
#
#   2. TRAIN_STRATEGY: Datos históricos divididos en:
#      - TRAIN: Conjunto de entrenamiento (años 2000-2018)
#      - VALIDATE: Conjunto de validación para ajustar hiperparámetros (año 2019)
#      - TEST: Conjunto de prueba para evaluar el modelo final (año 2020)
#
#   3. TRAIN_FINAL: Todos los datos disponibles con clase (años 2000-2020)
#                   para entrenar el modelo final después de encontrar
#                   los mejores hiperparámetros
#
# ¿POR QUÉ ES IMPORTANTE ESTA DIVISIÓN?
# - Evita OVERFITTING (que el modelo memorice en lugar de aprender)
# - Permite evaluar el modelo en datos nunca vistos (TEST)
# - Simula cómo funcionará el modelo en el futuro (2022)
# ============================================================================

# ----------------------------------------------------------------------------
# LIBRERÍAS NECESARIAS
# ----------------------------------------------------------------------------
require("data.table")  # Manejo eficiente de datos
require("yaml")        # Leer archivos de configuración YAML
library(lubridate)     # Manejo de fechas

# ----------------------------------------------------------------------------
# FUNCIÓN: APLICAR PARTICIÓN
# ----------------------------------------------------------------------------
# Esta función crea una columna binaria (0/1) indicando si cada registro
# pertenece o no a una sección específica (train, validate, test, etc.)
#
# ¿CÓMO FUNCIONA?
# 1. Crea una columna nueva llamada "part_[seccion]" (ej: part_train)
# 2. Marca con 1 los registros que cumplen los criterios de la sección
# 3. Marca con 0 los registros que NO cumplen los criterios
#
# Parámetros:
#   - seccion: Nombre de la sección a particionar (train, validate, test, etc.)
#
# Criterios de selección (según configuración YAML):
#   - periodos: Lista específica de años a incluir
#   - rango: Rango de años (desde-hasta) a incluir
#   - excluir: Lista de años a excluir explícitamente
#   - undersampling: Reducir aleatoriamente ciertos valores de clase

aplicar_particion <- function(seccion) {

  # Nombre de la nueva columna: part_train, part_validate, etc.
  columna_nueva <- paste0("part_", seccion)

  # Inicializar la columna en 0 (ningún registro pertenece inicialmente)
  dataset[, (columna_nueva) := 0L]

  # CRITERIO 1: Seleccionar por PERÍODOS ESPECÍFICOS
  # Si en el YAML hay una lista de períodos (ej: periodos: [2019, 2020])
  # se marcan con 1 los registros de esos años
  if (length(PARAMS$training_strategy$param[[seccion]]$periodos) > 0) {

    dataset[get(PARAMS$training_strategy$const$periodo) %in%
              PARAMS$training_strategy$param[[seccion]]$periodos,
            (columna_nueva) := 1L]

  } else {

    # CRITERIO 2: Seleccionar por RANGO DE AÑOS
    # Si no hay lista de períodos, usar rango (desde-hasta)
    # Ejemplo: desde: 2000, hasta: 2018
    dataset[get(PARAMS$training_strategy$const$periodo) >=
              PARAMS$training_strategy$param[[seccion]]$rango$desde &
            get(PARAMS$training_strategy$const$periodo) <=
              PARAMS$training_strategy$param[[seccion]]$rango$hasta,
            (columna_nueva) := 1L]
  }

  # CRITERIO 3: EXCLUIR años específicos
  # Si en el YAML hay una lista de años a excluir (ej: excluir: [2008, 2009])
  # se marcan con 0 esos años, incluso si fueron seleccionados antes
  if (length(PARAMS$training_strategy$param[[seccion]]$excluir) > 0) {

    dataset[get(PARAMS$training_strategy$const$periodo) %in%
              PARAMS$training_strategy$param[[seccion]]$excluir,
            (columna_nueva) := 0L]
  }

  # CRITERIO 4: UNDERSAMPLING (reducir cantidad de ciertos valores)
  # Útil en clasificación desbalanceada. En regresión típicamente no se usa.
  # Ejemplo: Si clase=100 y prob=0.3, solo el 30% de registros con clase=100
  #          se mantienen en la partición (los demás se marcan como 0)
  if ("undersampling" %in% names(PARAMS$training_strategy$param[[seccion]])) {

    for (clase_valor in PARAMS$training_strategy$param[[seccion]]$undersampling) {

      # part_azar es una variable aleatoria entre 0 y 1
      # Si part_azar > prob, el registro se excluye (se marca como 0)
      dataset[get(columna_nueva) == 1L &
              get(PARAMS$training_strategy$const$clase) == clase_valor$clase &
              part_azar > clase_valor$prob,
              (columna_nueva) := 0L]
    }
  }
}

# ============================================================================
# PROGRAMA PRINCIPAL - PIPELINE DE TRAINING STRATEGY
# ============================================================================
# A partir de aquí comienza la ejecución del script.
# El pipeline sigue estos pasos principales:
#
# 1. Cargar el dataset con Feature Engineering
# 2. Eliminar variables que no deben usarse como predictoras
# 3. Definir automáticamente los períodos de cada conjunto
# 4. Aplicar las particiones (train, validate, test, present, train_final)
# 5. Generar archivo de control para verificación
# 6. Guardar los datasets particionados
# ============================================================================

cat("\n=== INICIANDO TRAINING STRATEGY ===\n\n")

# ----------------------------------------------------------------------------
# PASO 1: CARGAR EL DATASET CON FEATURE ENGINEERING
# ----------------------------------------------------------------------------

# Establecer directorio base del proyecto
setwd(PARAMS$environment$base_dir)

# Construir nombres de carpetas y archivos basándose en configuración YAML
# Ejemplo: hf3_max_minimo
nom_exp_folder <- paste(PARAMS$experiment$experiment_label,
                        PARAMS$experiment$experiment_code, sep = "_")

# Ejemplo: hf3_max_minimo_f1 (f1 = forecast 1 año adelante)
nom_subexp_folder <- paste(PARAMS$experiment$experiment_label,
                           PARAMS$experiment$experiment_code,
                           paste0("f", PARAMS$feature_engineering$const$orden_lead),
                           sep = "_")

# Nombre del archivo de entrada (dataset con FE)
nom_arch <- paste0(paste(PARAMS$experiment$experiment_label,
                        PARAMS$experiment$experiment_code,
                        paste0("f", PARAMS$feature_engineering$const$orden_lead),
                        sep = "_"), ".csv.gz")

# Navegar al directorio donde está el dataset con Feature Engineering
setwd(paste0("./exp/", nom_exp_folder, "/", nom_subexp_folder, "/01_FE"))

# Cargar el dataset
cat("Cargando dataset desde:", nom_arch, "\n")
dataset <- fread(nom_arch)
cat("Dataset cargado:", nrow(dataset), "filas x", ncol(dataset), "columnas\n\n")

# ----------------------------------------------------------------------------
# PASO 2: ELIMINAR VARIABLES QUE NO DEBEN USARSE COMO PREDICTORAS
# ----------------------------------------------------------------------------
# Eliminamos todas las variables que comienzan con "hf3_" EXCEPTO la clase.
#
# ¿Por qué?
# Las variables hf3_* (como hf3_ppp_pc original, hf3_lag1, etc.) contienen
# información directa sobre lo que queremos predecir. Si las dejamos, el
# modelo "hará trampa" usando información futura.
#
# EXCEPCIÓN: La variable "clase" (hf3_ppp_pc del año siguiente) sí se mantiene
# porque es la variable objetivo que queremos predecir.

vars_eliminar <- setdiff(grep("hf3", names(dataset), value = TRUE),
                        PARAMS$feature_engineering$const$clase)

if (length(vars_eliminar) > 0) {
  dataset[, (vars_eliminar) := NULL]
  cat("Variables hf3_* eliminadas (excepto clase):", length(vars_eliminar), "\n")
  cat("Variables eliminadas:", paste(vars_eliminar, collapse = ", "), "\n\n")
}

# ----------------------------------------------------------------------------
# PASO 3: DEFINIR AUTOMÁTICAMENTE LOS PERÍODOS
# ----------------------------------------------------------------------------
# El script calcula automáticamente qué años usar para cada conjunto,
# basándose en el año presente y el orden de lead.
#
# LÓGICA:
# - present_year = 2021 (último año con datos, pero SIN clase)
# - max_year_with_clase = 2020 (último año donde conocemos la clase)
#
# PARTICIONES:
# - TEST: año 2020 (último año con clase disponible)
# - VALIDATE: año 2019 (año anterior al test)
# - TRAIN: años 2000-2018 (desde el inicio hasta 2 años antes del test)
# - TRAIN_FINAL: años 2000-2020 (todo el histórico con clase)
# - PRESENT: año 2021 (datos sin clase para hacer predicciones de 2022)

cat("=== CONFIGURACIÓN AUTOMÁTICA DE PERÍODOS ===\n\n")

# Año presente: último año disponible en el dataset (sin clase)
present_year <- PARAMS$feature_engineering$const$presente
cat("Año presente (sin clase):", present_year, "\n")

# Último año donde tenemos clase disponible
# Ejemplo: Si presente=2021 y orden_lead=1, entonces max_year_with_clase=2020
max_year_with_clase <- present_year - PARAMS$feature_engineering$const$orden_lead
cat("Último año con clase disponible:", max_year_with_clase, "\n\n")

# TEST: último año con clase disponible (2020)
PARAMS$training_strategy$param$test$periodos <- max_year_with_clase
cat("TEST configurado para año:", max_year_with_clase, "\n")

# VALIDATE: año anterior al test (2019)
PARAMS$training_strategy$param$validate$periodos <- max_year_with_clase - 1
cat("VALIDATE configurado para año:", max_year_with_clase - 1, "\n")

# TRAIN: hasta 2 años antes del test (2000-2018)
PARAMS$training_strategy$param$train$rango$hasta <- max_year_with_clase - 2
cat("TRAIN configurado desde:",
    PARAMS$training_strategy$param$train$rango$desde,
    "hasta:", max_year_with_clase - 2, "\n")

# TRAIN_FINAL: hasta el último año del test (2000-2020)
# Se usa después de la optimización de hiperparámetros para entrenar
# el modelo final con todos los datos disponibles
PARAMS$training_strategy$param$train_final$rango$hasta <- max_year_with_clase
cat("TRAIN_FINAL configurado desde:",
    PARAMS$training_strategy$param$train_final$rango$desde,
    "hasta:", max_year_with_clase, "\n")

# PRESENT: año sin clase para hacer predicciones (2021)
PARAMS$training_strategy$param$present$periodos <- present_year
cat("PRESENT configurado para año:", present_year, "en adelante\n\n")

cat("=== FIN DE CONFIGURACIÓN AUTOMÁTICA ===\n\n")

# ----------------------------------------------------------------------------
# PASO 4: APLICAR LAS PARTICIONES
# ----------------------------------------------------------------------------
# Ahora vamos a crear las columnas part_train, part_validate, part_test, etc.
# Cada columna es binaria (0/1) indicando si el registro pertenece a esa partición.

# Ordenar el dataset por Country Code y year
setorderv(dataset, PARAMS$training_strategy$const$campos_sort)

# Establecer semilla para reproducibilidad
set.seed(PARAMS$training_strategy$param$semilla)

# Crear variable aleatoria entre 0 y 1 para undersampling (si se usa)
# Esta variable se usa en aplicar_particion() para seleccionar
# aleatoriamente qué registros mantener cuando hay undersampling
dataset[, part_azar := runif(nrow(dataset))]

# Aplicar particiones para cada sección
# Las secciones son: present, train, validate, test, train_final
cat("Aplicando particiones...\n")
for (seccion in PARAMS$training_strategy$const$secciones) {
  cat("  - Procesando:", seccion, "\n")
  aplicar_particion(seccion)
}

# Ya no necesitamos la variable aleatoria
dataset[, part_azar := NULL]
cat("\n")

# ----------------------------------------------------------------------------
# PASO 5: GENERAR ARCHIVO DE CONTROL
# ----------------------------------------------------------------------------
# El archivo de control muestra cuántos registros hay en cada combinación
# de particiones. DEBE SER REVISADO para verificar que las particiones
# son correctas.
#
# Ejemplo de salida:
# part_present | part_train | part_validate | part_test | part_train_final | N
# 0            | 1          | 0             | 0         | 1                | 5000
# 0            | 0          | 1             | 0         | 1                | 500
# 0            | 0          | 0             | 1         | 1                | 500
# 1            | 0          | 0             | 0         | 0                | 200

psecciones <- paste0("part_", PARAMS$training_strategy$const$secciones)

# Contar registros por combinación de particiones
tb_control <- dataset[, .N, psecciones]

cat("=== RESUMEN DE PARTICIONES ===\n")
print(tb_control)
cat("\n")

# Verificar que cada partición crítica tiene registros
cat("=== VERIFICACIÓN DE PARTICIONES ===\n")
cat("Present (sin clase):", dataset[part_present > 0, .N], "registros\n")
cat("Train:", dataset[part_train > 0, .N], "registros\n")
cat("Validate:", dataset[part_validate > 0, .N], "registros\n")
cat("Test:", dataset[part_test > 0, .N], "registros\n")
cat("Train_final:", dataset[part_train_final > 0, .N], "registros\n\n")

# Advertencias si alguna partición está vacía
if (dataset[part_train > 0, .N] == 0) {
  cat("ADVERTENCIA: La partición TRAIN está vacía!\n")
}
if (dataset[part_validate > 0, .N] == 0) {
  cat("ADVERTENCIA: La partición VALIDATE está vacía!\n")
}
if (dataset[part_test > 0, .N] == 0) {
  cat("ADVERTENCIA: La partición TEST está vacía!\n")
}

# ----------------------------------------------------------------------------
# PASO 6: GUARDAR LOS DATASETS PARTICIONADOS
# ----------------------------------------------------------------------------

# Crear directorio de salida (02_TS)
setwd("../")
dir.create("02_TS", showWarnings = FALSE)
setwd("02_TS")

cat("\n=== GUARDANDO ARCHIVOS ===\n")

# --- GUARDAR ARCHIVO DE CONTROL ---
# Este archivo de texto muestra el resumen de particiones
fwrite(tb_control,
       file = paste0(PARAMS$training_strategy$files$output$control),
       sep = "\t")

cat("1. Archivo de control guardado:",
    PARAMS$training_strategy$files$output$control, "\n")

# --- GUARDAR PRESENT DATA ---
# Contiene los datos del año 2021 (sin clase) para hacer predicciones de 2022.
# NO incluye la columna "clase" porque no la conocemos (es el futuro).
# NO incluye las columnas part_* porque no son necesarias.
if (0 < dataset[part_present > 0, .N]) {

  fwrite(dataset[part_present > 0,
                 setdiff(colnames(dataset),
                        c(psecciones, PARAMS$training_strategy$const$clase)),
                 with = FALSE],
         file = paste0(PARAMS$training_strategy$files$output$present_data),
         logical01 = TRUE,
         sep = ",")

  cat("2. Present data guardado:",
      PARAMS$training_strategy$files$output$present_data,
      "con", dataset[part_present > 0, .N], "registros\n")

} else {
  cat("2. Present data: No hay registros para guardar\n")
}

# --- GUARDAR TRAIN STRATEGY ---
# Contiene los registros de train, validate y test con sus etiquetas de partición.
# Incluye las columnas part_train, part_validate, part_test para identificar
# a qué conjunto pertenece cada registro.
# NO incluye part_present ni part_train_final porque no son necesarias aquí.
# SÍ incluye la columna "clase" porque es necesaria para entrenar y evaluar.
if (0 < dataset[part_train > 0 | part_validate > 0 | part_test > 0, .N]) {

  fwrite(dataset[part_train > 0 | part_validate > 0 | part_test > 0,
                 setdiff(colnames(dataset),
                        c("part_present", "part_train_final")),
                 with = FALSE],
         file = paste0(PARAMS$training_strategy$files$output$train_strategy),
         logical01 = TRUE,
         sep = ",")

  cat("3. Train strategy guardado:",
      PARAMS$training_strategy$files$output$train_strategy,
      "con", dataset[part_train > 0 | part_validate > 0 | part_test > 0, .N],
      "registros\n")

} else {
  cat("3. Train strategy: No hay registros para guardar\n")
}

# --- GUARDAR TRAIN FINAL ---
# Contiene todos los datos históricos con clase (2000-2020) para entrenar
# el modelo final DESPUÉS de encontrar los mejores hiperparámetros.
# NO incluye columnas part_* porque este dataset se usa completo.
# SÍ incluye la columna "clase".
if (0 < dataset[part_train_final > 0, .N]) {

  fwrite(dataset[part_train_final > 0,
                 setdiff(colnames(dataset), psecciones),
                 with = FALSE],
         file = paste0(PARAMS$training_strategy$files$output$train_final),
         logical01 = TRUE,
         sep = ",")

  cat("4. Train final guardado:",
      PARAMS$training_strategy$files$output$train_final,
      "con", dataset[part_train_final > 0, .N], "registros\n")

} else {
  cat("4. Train final: No hay registros para guardar\n")
}

# ----------------------------------------------------------------------------
# RESUMEN FINAL
# ----------------------------------------------------------------------------
cat("\n=== TRAINING STRATEGY COMPLETADO ===\n")
cat("Archivos generados en directorio: 02_TS\n\n")
cat("IMPORTANTE: Revise el archivo '", PARAMS$training_strategy$files$output$control,
    "' para verificar que las particiones son correctas.\n\n", sep = "")

cat("Próximos pasos:\n")
cat("1. Abrir y revisar el archivo de control (",
    PARAMS$training_strategy$files$output$control, ")\n", sep = "")
cat("2. Verificar que las cantidades de registros son razonables\n")
cat("3. Ejecutar el script 03_HT_health.R para optimizar hiperparámetros\n\n")

cat("ARCHIVOS GENERADOS:\n")
cat("  -", PARAMS$training_strategy$files$output$control,
    "(archivo de control - REVISAR SIEMPRE)\n")
cat("  -", PARAMS$training_strategy$files$output$present_data,
    "(datos sin clase para predicción)\n")
cat("  -", PARAMS$training_strategy$files$output$train_strategy,
    "(train + validate + test con particiones)\n")
cat("  -", PARAMS$training_strategy$files$output$train_final,
    "(todos los datos para modelo final)\n\n")

cat("Fin del script. ¡Buena suerte con la optimización!\n")

