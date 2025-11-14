################################################################################
# TRAINING STRATEGY - Definición de Períodos Train/Validate/Test
################################################################################
#
# Este script toma el dataset con Feature Engineering y lo particiona en:
#   1. Present: Datos SIN clase (año 2022) donde se aplicará el modelo
#   2. Train/Validate/Test: Datos CON clase para entrenar y evaluar
#   3. Train_final: Datos para entrenar el modelo final con mejores hiperparámetros
#
# IMPORTANTE: 
#   - Los períodos se calculan AUTOMÁTICAMENTE según 'presente' y 'orden_lead'
#   - Puedes EXCLUIR años en la sección 'train' del YML (ej: COVID 2020-2021)
#   - El script usa 'year' como período temporal (no 'foto_mes')
#
################################################################################

require("data.table")
require("yaml")
library(lubridate)

################################################################################
# FUNCIÓN: Aplicar Partición
################################################################################
# Crea una variable binaria (0/1) que indica si un registro pertenece a una
# sección específica (train, validate, test, present, train_final).
#
# La lógica permite:
#   - Especificar períodos individuales (ej: 2020, 2021)
#   - Especificar rangos (ej: desde=2000, hasta=2019)
#   - Excluir períodos específicos (ej: excluir=[2020, 2021])
#   - Hacer undersampling de ciertas clases (balanceo de datos)

aplicar_particion <- function(seccion) {
  columna_nueva <- paste0("part_", seccion)
  dataset[, (columna_nueva) := 0L]

  # Si hay períodos específicos listados, usar esos
  if (length(PARAMS$training_strategy$param[[seccion]]$periodos) > 0) {
    dataset[get(PARAMS$training_strategy$const$periodo) %in% PARAMS$training_strategy$param[[seccion]]$periodos,
      (columna_nueva) := 1L
    ]
  } else {
    # Si no, usar rango desde-hasta
    dataset[get(PARAMS$training_strategy$const$periodo) >= PARAMS$training_strategy$param[[seccion]]$rango$desde &
      get(PARAMS$training_strategy$const$periodo) <= PARAMS$training_strategy$param[[seccion]]$rango$hasta,
    (columna_nueva) := 1L
    ]
  }

  # Excluir períodos específicos si están definidos
  if (length(PARAMS$training_strategy$param[[seccion]]$excluir) > 0) {
    dataset[get(PARAMS$training_strategy$const$periodo) %in% PARAMS$training_strategy$param[[seccion]]$excluir,
      (columna_nueva) := 0L
    ]
  }

  # Undersampling: reducir proporción de ciertas clases si está configurado
  if ("undersampling" %in% names(PARAMS$training_strategy$param[[seccion]])) {
    for (clase_valor in PARAMS$training_strategy$param[[seccion]]$undersampling) {
      dataset[get(columna_nueva) == 1L &
        get(PARAMS$training_strategy$const$clase) == clase_valor$clase &
        part_azar > clase_valor$prob,
      (columna_nueva) := 0L
      ]
    }
  }
}

################################################################################
# INICIO DEL PROGRAMA
################################################################################

cat("\n")
cat("================================================================================\n")
cat("  TRAINING STRATEGY - HEALTH ECONOMICS\n")
cat("================================================================================\n")
cat("\n")

# Establecer directorio base
setwd(PARAMS$environment$base_dir)

################################################################################
# CARGAR DATASET POST-FEATURE ENGINEERING
################################################################################

# Construir nombres de carpetas y archivos
nom_exp_folder <- paste(PARAMS$experiment$experiment_label, PARAMS$experiment$experiment_code, sep = "_")

nom_subexp_folder <- paste(PARAMS$experiment$experiment_label, PARAMS$experiment$experiment_code, paste0("f", PARAMS$feature_engineering$const$orden_lead), sep = "_")

nom_arch <- paste0(paste(PARAMS$experiment$experiment_label, PARAMS$experiment$experiment_code, paste0("f", PARAMS$feature_engineering$const$orden_lead), sep = "_"), ".csv.gz")

# Navegar al directorio del dataset
setwd(paste0("./exp/", nom_exp_folder, "/", nom_subexp_folder, "/01_FE"))

cat("Cargando dataset desde:", nom_arch, "\n")
dataset <- fread(nom_arch)

cat("Dataset cargado:", nrow(dataset), "filas x", ncol(dataset), "columnas\n")
cat("\n")

################################################################################
# ELIMINAR VARIABLES CON "hf3" (PREVENCIÓN DE DATA LEAKAGE)
################################################################################
# CRÍTICO: Eliminamos TODAS las variables que contengan "hf3" excepto 'clase'
# Esto previene data leakage de otras variables relacionadas con gasto en salud

cat("Eliminando variables con 'hf3' (prevención de data leakage)...\n")
vars_antes <- ncol(dataset)

# Mantener solo las columnas que NO contengan "hf3" en su nombre
dataset <- dataset[, !grepl("hf3", names(dataset)), with = FALSE]

vars_despues <- ncol(dataset)
cat("Variables eliminadas:", vars_antes - vars_despues, "\n")
cat("Variables restantes:", vars_despues, "\n")
cat("\n")

################################################################################
# CONFIGURACIÓN AUTOMÁTICA DE PERÍODOS
################################################################################
# Basándose en 'presente' y 'orden_lead' del YML, calculamos automáticamente:
#   - Test: Último año con clase disponible
#   - Validate: Año anterior al test
#   - Train: Hasta 2 años antes del test
#   - Train_final: Incluye test también para entrenamiento final del modelo

present_year <- PARAMS$feature_engineering$const$presente
max_year_with_clase <- present_year + PARAMS$feature_engineering$const$orden_lead

# Test: último año con clase disponible
PARAMS$training_strategy$param$test$periodos <- max_year_with_clase

# Validate: año anterior al test
PARAMS$training_strategy$param$validate$periodos <- max_year_with_clase - 1

# Train: hasta 2 años antes del test
PARAMS$training_strategy$param$train$rango$hasta <- max_year_with_clase - 2

# Train_final: incluye test también para entrenamiento final
PARAMS$training_strategy$param$train_final$rango$hasta <- max_year_with_clase

cat("================================================================================\n")
cat("  CONFIGURACIÓN AUTOMÁTICA DE PERÍODOS\n")
cat("================================================================================\n")
cat("\n")
cat("Basándose en:\n")
cat("  - presente:", present_year, "\n")
cat("  - orden_lead:", PARAMS$feature_engineering$const$orden_lead, "\n")
cat("\n")
cat("Se calcularon los siguientes períodos:\n")
cat("  - Train hasta:", PARAMS$training_strategy$param$train$rango$hasta, "\n")
cat("  - Validate:", PARAMS$training_strategy$param$validate$periodos, "\n")
cat("  - Test:", PARAMS$training_strategy$param$test$periodos, "\n")
cat("  - Present (años sin clase):", present_year, "en adelante\n")
cat("  - Train_final hasta:", PARAMS$training_strategy$param$train_final$rango$hasta, "\n")
cat("\n")

# ADVERTENCIA PEDAGÓGICA sobre exclusión de años
if (length(PARAMS$training_strategy$param$train$excluir) > 0) {
  cat("⚠️  ADVERTENCIA: Años EXCLUIDOS de training:\n")
  cat("   ", PARAMS$training_strategy$param$train$excluir, "\n")
  cat("    (Esto es parte de tu decisión estratégica sobre uso de datos COVID)\n")
  cat("\n")
} else {
  cat("✓  No hay años excluidos de training.\n")
  cat("   Estás usando TODOS los datos disponibles, incluyendo COVID 2020-2021.\n")
  cat("\n")
}

################################################################################
# APLICAR PARTICIONES
################################################################################

cat("================================================================================\n")
cat("  APLICANDO PARTICIONES\n")
cat("================================================================================\n")
cat("\n")

# Ordenar dataset por año y país
setorderv(dataset, PARAMS$training_strategy$const$campos_sort)

# Generar variable aleatoria para undersampling (si se usa)
set.seed(PARAMS$training_strategy$param$semilla)
dataset[, part_azar := runif(nrow(dataset))]

# Aplicar particiones para cada sección
for (seccion in PARAMS$training_strategy$const$secciones) {
  aplicar_particion(seccion)
}

# Ya no necesito la variable aleatoria
dataset[, part_azar := NULL]

################################################################################
# GENERAR ARCHIVO DE CONTROL
################################################################################

psecciones <- paste0("part_", PARAMS$training_strategy$const$secciones)

# Tabla de control: cantidad de registros en cada combinación de particiones
tb_control <- dataset[, .N, psecciones]

cat("Resumen de particiones generado.\n")
cat("\n")

# Mostrar resumen
cat("================================================================================\n")
cat("  RESUMEN DE PARTICIONES\n")
cat("================================================================================\n")
cat("\n")
print(tb_control)
cat("\n")

# Verificación de conteos
cat("================================================================================\n")
cat("  VERIFICACIÓN DE PARTICIONES\n")
cat("================================================================================\n")
cat("\n")
cat("Present (sin clase):", dataset[part_present > 0, .N], "registros\n")
cat("Train:", dataset[part_train > 0, .N], "registros\n")
cat("Validate:", dataset[part_validate > 0, .N], "registros\n")
cat("Test:", dataset[part_test > 0, .N], "registros\n")
cat("Train_final:", dataset[part_train_final > 0, .N], "registros\n")
cat("\n")

# ADVERTENCIAS PEDAGÓGICAS
if (dataset[part_train > 0, .N] < 100) {
  cat("⚠️  ADVERTENCIA: Muy pocos registros en TRAIN (", dataset[part_train > 0, .N], ").\n")
  cat("    Verifica tu configuración de 'presente' y 'orden_lead'.\n")
  cat("\n")
}

if (dataset[part_present > 0, .N] == 0) {
  cat("⚠️  ADVERTENCIA: No hay registros en PRESENT (datos sin clase).\n")
  cat("    Esto significa que no hay datos para predecir 2022.\n")
  cat("    Verifica que tu dataset incluya años hasta", present_year, ".\n")
  cat("\n")
}

################################################################################
# GUARDAR ARCHIVOS DE SALIDA
################################################################################

cat("================================================================================\n")
cat("  GUARDANDO ARCHIVOS\n")
cat("================================================================================\n")
cat("\n")

# Crear directorio de salida
setwd("../")
dir.create("02_TS", showWarnings = FALSE)
setwd("02_TS")

# 1. Guardar archivo de control
fwrite(tb_control,
  file = paste0(PARAMS$training_strategy$files$output$control),
  sep = "\t"
)

cat("✓ Archivo de control guardado:", PARAMS$training_strategy$files$output$control, "\n")

# 2. Guardar present (datos sin clase para predicción de 2022)
if (0 < dataset[part_present > 0, .N]) {
  fwrite(dataset[part_present > 0,
    setdiff(colnames(dataset), c(psecciones, PARAMS$training_strategy$const$clase)),
    with = FALSE
  ],
  file = paste0(PARAMS$training_strategy$files$output$present_data),
  logical01 = TRUE,
  sep = ","
  )

  cat("✓ Present data guardado:", PARAMS$training_strategy$files$output$present_data,
    "con", dataset[part_present > 0, .N], "registros\n"
  )
}

# 3. Guardar train_strategy (train + validate + test)
if (0 < dataset[part_train > 0 | part_validate > 0 | part_test > 0, .N]) {
  fwrite(dataset[part_train > 0 | part_validate > 0 | part_test > 0,
    setdiff(colnames(dataset), c("part_present", "part_train_final")),
    with = FALSE
  ],
  file = paste0(PARAMS$training_strategy$files$output$train_strategy),
  logical01 = TRUE,
  sep = ","
  )

  cat("✓ Train strategy guardado:", PARAMS$training_strategy$files$output$train_strategy,
    "con", dataset[part_train > 0 | part_validate > 0 | part_test > 0, .N], "registros\n"
  )
}

# 4. Guardar train_final (para entrenamiento final del modelo)
if (0 < dataset[part_train_final > 0, .N]) {
  fwrite(dataset[part_train_final > 0,
    setdiff(colnames(dataset), psecciones),
    with = FALSE
  ],
  file = paste0(PARAMS$training_strategy$files$output$train_final),
  logical01 = TRUE,
  sep = ","
  )

  cat("✓ Train final guardado:", PARAMS$training_strategy$files$output$train_final,
    "con", dataset[part_train_final > 0, .N], "registros\n"
  )
}

cat("\n")
cat("================================================================================\n")
cat("  TRAINING STRATEGY COMPLETADO EXITOSAMENTE\n")
cat("================================================================================\n")
cat("\n")
cat("Archivos generados en directorio: 02_TS\n")
cat("\n")
cat("IMPORTANTE: Revisa el archivo de control (", PARAMS$training_strategy$files$output$control, ")\n")
cat("para verificar que las particiones son correctas antes de continuar.\n")
cat("\n")
cat("Próximo paso: Ejecutar 03_HT_health.R para optimizar hiperparámetros.\n")
cat("\n")
