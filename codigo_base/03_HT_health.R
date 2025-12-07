# ============================================================================
# HYPERPARAMETER TUNING (OPTIMIZACIÓN DE HIPERPARÁMETROS) PARA HEALTH ECONOMICS
# Universidad del Oeste (UNO) - Aplicaciones en Ciencia de Datos
# ============================================================================
# Este script implementa la optimización de hiperparámetros usando
# Optimización Bayesiana para encontrar la mejor configuración de LightGBM
# que minimice el error de predicción (RMSE).
#
# ¿QUÉ HACE ESTE SCRIPT?
# 1. Carga los datos con training strategy (train/validate/test)
# 2. Define el espacio de búsqueda de hiperparámetros (rangos a explorar)
# 3. Usa Optimización Bayesiana para encontrar la mejor combinación
# 4. Entrena el modelo final con los mejores hiperparámetros
# 5. Genera predicciones para el año 2022
#
# ¿QUÉ ES LA OPTIMIZACIÓN BAYESIANA?
# Es un método inteligente para encontrar los mejores hiperparámetros.
# En lugar de probar combinaciones al azar, aprende de cada intento y
# decide estratégicamente qué combinación probar a continuación.
# Es como un científico que diseña experimentos de forma cada vez más inteligente.
#
# ANALOGÍA: Si tuvieras que encontrar el punto más alto de una montaña con
# niebla (no ves todo), la BO te ayuda a decidir qué dirección tomar en cada
# paso basándose en lo que ya exploraste.
# ============================================================================

# ----------------------------------------------------------------------------
# LIBRERÍAS NECESARIAS
# ----------------------------------------------------------------------------
require("data.table")   # Manejo eficiente de datos
require("primes")       # Números primos (usado por algunas funciones internas)
require("lightgbm")     # Algoritmo de Machine Learning

# Paquetes para Optimización Bayesiana
require("DiceKriging") # Kriging para modelar la superficie de búsqueda
require("mlrMBO")      # Machine Learning R - Model-Based Optimization
require("rlist")       # Manejo de listas

# Configurar uso de threads (65% de los cores disponibles)
setDTthreads(percent = 65)

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

# ----------------------------------------------------------------------------
# FUNCIÓN: LOGUEAR
# ----------------------------------------------------------------------------
# Guarda los resultados de cada iteración en un archivo de texto.
# Esto permite:
# - Trackear el progreso de la optimización
# - Retomar si se interrumpe el proceso
# - Analizar qué hiperparámetros funcionaron mejor
#
# Parámetros:
#   - reg: Lista con los valores a guardar (hiperparámetros, error, etc.)
#   - arch: Nombre del archivo (si es NA, se genera automáticamente)
#   - folder: Carpeta donde guardar
#   - ext: Extensión del archivo
#   - verbose: Si TRUE, también imprime en pantalla

loguear <- function(reg, arch = NA, folder = "./exp/", ext = ".txt", verbose = TRUE) {

  # Determinar nombre del archivo
  archivo <- arch
  if (is.na(arch)) archivo <- paste0(folder, substitute(reg), ext)

  # Si el archivo no existe, crear y escribir encabezados
  if (!file.exists(archivo)) {
    linea <- paste0("fecha\t",
                    paste(list.names(reg), collapse = "\t"), "\n")
    cat(linea, file = archivo)
  }

  # Agregar nueva línea con fecha/hora y valores
  linea <- paste0(format(Sys.time(), "%Y%m%d %H%M%S"), "\t", # timestamp
                  gsub(", ", "\t", toString(reg)), "\n")

  cat(linea, file = archivo, append = TRUE) # Guardar en archivo

  if (verbose) cat(linea) # Imprimir en pantalla
}

# ----------------------------------------------------------------------------
# FUNCIÓN: PARAMETRIZAR
# ----------------------------------------------------------------------------
# Separa los hiperparámetros en dos grupos:
# 1. Fijos: valores únicos que no se optimizan (ej: seed = 999983)
# 2. Variables: rangos a explorar (ej: learning_rate entre 0.01 y 0.2)
#
# ¿Por qué es útil?
# Algunos hiperparámetros sabemos qué valor usar (ej: objetivo = "regression").
# Otros queremos que la BO los explore (ej: ¿learning_rate óptimo?).
#
# Parámetros:
#   - lparam: Lista de hiperparámetros desde el YAML
#
# Retorna:
#   - param_fijos: Hiperparámetros con valores únicos
#   - paramSet: Hiperparámetros a optimizar (con sus rangos)

parametrizar <- function(lparam) {
  param_fijos <- copy(lparam)
  hs <- list()

  # Recorrer cada hiperparámetro
  for (param in names(lparam)) {

    # Si tiene más de un valor, es un rango a optimizar
    if (length(lparam[[param]]) > 1) {
      desde <- as.numeric(lparam[[param]][[1]])
      hasta <- as.numeric(lparam[[param]][[2]])

      # Si tiene 2 elementos: parámetro CONTINUO (ej: 0.01 a 0.2)
      if (length(lparam[[param]]) == 2) {
        hs <- append(hs,
                     list(makeNumericParam(param, lower = desde, upper = hasta)))
      } else {
        # Si tiene 3 elementos: parámetro ENTERO (ej: 10 a 1500, paso 1)
        hs <- append(hs,
                     list(makeIntegerParam(param, lower = desde, upper = hasta)))
      }

      # Quitar de param_fijos porque es variable
      param_fijos[[param]] <- NULL
    }
  }

  return(list("param_fijos" = param_fijos,
              "paramSet" = hs))
}

# ----------------------------------------------------------------------------
# FUNCIÓN: PARTICIONAR
# ----------------------------------------------------------------------------
# Divide un dataset en K folds (partes) de forma estratificada.
# Estratificado = mantiene la proporción de cada grupo en cada fold.
#
# Ejemplo: Si divides en 2 partes iguales, cada parte tendrá aproximadamente
# la misma distribución de países y años.
#
# Parámetros:
#   - data: Dataset a particionar
#   - division: Vector indicando tamaño de cada fold (ej: c(1,1) = 2 partes iguales)
#   - agrupa: Columnas por las cuales estratificar
#   - campo: Nombre de la columna donde guardar el fold asignado
#   - start: Número inicial de fold
#   - seed: Semilla para reproducibilidad

particionar <- function(data, division, agrupa = "", campo = "fold", start = 1, seed = NA) {
  if (!is.na(seed)) set.seed(seed)

  # Crear vector de folds repetidos
  bloque <- unlist(mapply(function(x, y) { rep(y, x) },
                         division,
                         seq(from = start, length.out = length(division))))

  # Asignar folds aleatoriamente, estratificado por grupos
  data[, (campo) := sample(rep(bloque, ceiling(.N / length(bloque))))[1:.N],
       by = agrupa]
}

# ============================================================================
# FUNCIÓN PRINCIPAL: ESTIMAR GANANCIA (ERROR) DE LIGHTGBM
# ============================================================================
# Esta es la función que la Optimización Bayesiana llama en cada iteración.
# Recibe una combinación de hiperparámetros, entrena un modelo, y retorna
# el error (RMSE) en el conjunto de test.
#
# ¿Cómo funciona el proceso?
# 1. BO propone una combinación de hiperparámetros (ej: learning_rate=0.05, num_leaves=100)
# 2. Esta función entrena un modelo con esos hiperparámetros
# 3. Evalúa el modelo en test y calcula el error (RMSE)
# 4. Retorna el error a BO
# 5. BO usa este resultado para decidir qué probar a continuación
#
# Parámetros:
#   - x: Lista con los hiperparámetros variables propuestos por BO

EstimarGanancia_lightgbm <- function(x) {
  gc()  # Liberar memoria

  # Incrementar contador de iteraciones
  GLOBAL_iteracion <<- GLOBAL_iteracion + 1

  # Combinar hiperparámetros fijos + variables
  param_completo <- c(param_fijos, x)

  # Configurar número de iteraciones y early stopping
  # Early stopping = detener entrenamiento si no hay mejora
  # El número de rondas depende del learning_rate:
  # learning_rate bajo = necesita más iteraciones para aprender
  param_completo$num_iterations <- ifelse(param_fijos$boosting == "dart", 999, 99999)
  param_completo$early_stopping_rounds <- as.integer(200 + 4 / param_completo$learning_rate)

  # Entrenar modelo con estos hiperparámetros
  set.seed(param_completo$seed)
  modelo_train <- lgb.train(
    data = dtrain,
    valids = list(valid = dvalidate),
    param = param_completo,
    verbose = -100  # Silencioso (no mostrar progreso)
  )

  # ----------------------------------------------------------------------------
  # EVALUAR EL MODELO EN TEST
  # ----------------------------------------------------------------------------
  # Aplicar el modelo al conjunto de test (datos nunca vistos)
  prediccion <- predict(modelo_train,
                       data.matrix(dataset_test[, campos_buenos, with = FALSE]))

  # Crear tabla con valores reales y predicciones
  tbl <- dataset_test[, c(PARAMS$hyperparameter_tuning$const$campo_clase), with = F]
  tbl[, pred := prediccion]

  gc()

  # Extraer la métrica del modelo (RMSE por defecto)
  # El modelo guarda en cada iteración el error en validation
  # Tomamos el error de la mejor iteración (best_iter)
  parametro <- unlist(modelo_train$record_evals$valid[[PARAMS$hyperparameter_tuning$param$lightgbm$metric]]$eval)[modelo_train$best_iter]

  ganancia_test_normalizada <- parametro

  rm(tbl)
  gc()

  # ----------------------------------------------------------------------------
  # GUARDAR IMPORTANCIA DE VARIABLES SI ES EL MEJOR MODELO HASTA AHORA
  # ----------------------------------------------------------------------------
  # La importancia de variables indica qué variables son más útiles
  # para hacer predicciones. Esto ayuda a entender el modelo.

  if (ganancia_test_normalizada < GLOBAL_ganancia) {
    GLOBAL_ganancia <<- ganancia_test_normalizada
    tb_importancia <- as.data.table(lgb.importance(modelo_train))

    # Guardar archivo con número de iteración
    fwrite(tb_importancia,
           file = paste0(PARAMS$hyperparameter_tuning$files$output$importancia,
                        GLOBAL_iteracion, ".txt"),
           sep = "\t")
  }

  # ----------------------------------------------------------------------------
  # LOGUEAR RESULTADOS DE ESTA ITERACIÓN
  # ----------------------------------------------------------------------------
  # Guardar en el archivo de log: hiperparámetros usados + error obtenido

  ds <- list("cols" = ncol(dtrain), "rows" = nrow(dtrain))
  xx <- c(ds, copy(param_completo))

  xx$early_stopping_rounds <- NULL
  xx$num_iterations <- modelo_train$best_iter  # Iteraciones reales usadas
  xx$ganancia <- ganancia_test_normalizada     # Error (RMSE)
  xx$iteracion_bayesiana <- GLOBAL_iteracion

  loguear(xx, arch = PARAMS$hyperparameter_tuning$files$output$BOlog)

  # Retornar el error a la Optimización Bayesiana
  return(ganancia_test_normalizada)
}

# ============================================================================
# PROGRAMA PRINCIPAL - PIPELINE DE HYPERPARAMETER TUNING
# ============================================================================
# A partir de aquí comienza la ejecución del script.
# El pipeline sigue estos pasos principales:
#
# 1. Cargar datos con training strategy
# 2. Preparar datasets para LightGBM (train, validate, test)
# 3. Configurar la Optimización Bayesiana
# 4. Ejecutar la optimización (encontrar mejores hiperparámetros)
# 5. Entrenar modelo final con mejores hiperparámetros
# 6. Generar predicciones para 2022
# ============================================================================

cat("\n=== INICIANDO HYPERPARAMETER TUNING ===\n\n")

# Establecer semilla para reproducibilidad
set.seed(PARAMS$hyperparameter_tuning$param$semilla)

# Navegar al directorio correcto
setwd(paste0(carpeta_base, "/exp"))
setwd(experiment_dir)
setwd(experiment_lead_dir)
setwd("02_TS")

# ----------------------------------------------------------------------------
# PASO 1: CARGAR DATASET CON TRAINING STRATEGY
# ----------------------------------------------------------------------------
cat("Paso 1: Cargando dataset con training strategy...\n")
nom_arch <- PARAMS$hyperparameter_tuning$files$input$dentrada
dataset <- fread(nom_arch)
cat("Dataset cargado:", nrow(dataset), "filas x", ncol(dataset), "columnas\n\n")

# Crear carpeta para resultados de Hyperparameter Tuning
setwd(paste0(carpeta_base, "/exp"))
setwd(experiment_dir)
setwd(experiment_lead_dir)
dir.create("03_HT", showWarnings = FALSE)
setwd("03_HT")

# ----------------------------------------------------------------------------
# PASO 2: PREPARAR DATASETS PARA LIGHTGBM
# ----------------------------------------------------------------------------
cat("Paso 2: Preparando datasets para LightGBM...\n")

# Identificar campos útiles (excluir clase y columnas de partición)
campos_buenos <- setdiff(copy(colnames(dataset)),
                        c(PARAMS$hyperparameter_tuning$const$campo_clase,
                          "part_train", "part_validate", "part_test"))

cat("Variables predictoras:", length(campos_buenos), "\n")

# --- CREAR DATASET DE ENTRENAMIENTO (TRAIN) ---
# Este dataset se usa para entrenar el modelo en cada iteración de BO
dtrain <- lgb.Dataset(
  data = data.matrix(dataset[part_train == 1, campos_buenos, with = FALSE]),
  label = dataset[part_train == 1][[PARAMS$hyperparameter_tuning$const$campo_clase]],
  free_raw_data = FALSE
)

cat("Train:", nrow(dataset[part_train == 1]), "registros\n")

# --- CREAR DATASETS DE VALIDACIÓN Y TEST ---
# Validación: para early stopping durante el entrenamiento
# Test: para evaluar el modelo y comparar hiperparámetros
#
# Dos estrategias posibles:
# A) validate=TRUE: usar validate para early stopping, test para evaluar
# B) validate=FALSE: dividir test en dos mitades (una para validate, otra para test)

if (PARAMS$hyperparameter_tuning$param$crossvalidation == FALSE) {

  if (PARAMS$hyperparameter_tuning$param$validate == TRUE) {
    # ESTRATEGIA A: Usar validate dedicado

    dvalidate <- lgb.Dataset(
      data = data.matrix(dataset[part_validate == 1, campos_buenos, with = FALSE]),
      label = dataset[part_validate == 1][[PARAMS$hyperparameter_tuning$const$campo_clase]],
      free_raw_data = FALSE
    )

    dataset_test <- dataset[part_test == 1]
    test_multiplicador <- 1

    cat("Validate:", nrow(dataset[part_validate == 1]), "registros\n")
    cat("Test:", nrow(dataset_test), "registros\n")

  } else {
    # ESTRATEGIA B: Dividir test en dos mitades

    cat("Dividiendo test en validate y test...\n")

    particionar(dataset,
                division = c(1, 1),  # Dos partes iguales
                agrupa = c("part_test", PARAMS$hyperparameter_tuning$const$campo_periodo),
                seed = PARAMS$hyperparameter_tuning$param$semilla,
                campo = "fold_test")

    # fold_test==1 para validation
    dvalidate <- lgb.Dataset(
      data = data.matrix(dataset[part_test == 1 & fold_test == 1, campos_buenos, with = FALSE]),
      label = dataset[part_test == 1 & fold_test == 1, PARAMS$hyperparameter_tuning$const$campo_clase],
      free_raw_data = FALSE
    )

    # fold_test==2 para test
    dataset_test <- dataset[part_test == 1 & fold_test == 2, ]
    test_multiplicador <- 2

    cat("Validate (mitad de test):", nrow(dataset[part_test == 1 & fold_test == 1]), "registros\n")
    cat("Test (mitad de test):", nrow(dataset_test), "registros\n")
  }
}

rm(dataset)
gc()
cat("\n")

# ----------------------------------------------------------------------------
# PASO 3: PREPARAR OPTIMIZACIÓN BAYESIANA
# ----------------------------------------------------------------------------
cat("Paso 3: Configurando Optimización Bayesiana...\n")

# Separar hiperparámetros fijos vs variables
hiperparametros <- PARAMS$hyperparameter_tuning$param[[PARAMS$hyperparameter_tuning$param$algoritmo]]
apertura <- parametrizar(hiperparametros)
param_fijos <- apertura$param_fijos

cat("Hiperparámetros fijos:", length(param_fijos), "\n")
cat("Hiperparámetros a optimizar:", length(apertura$paramSet), "\n")

# Mostrar qué hiperparámetros se van a optimizar
cat("\nHiperparámetros a optimizar:\n")
for (i in seq_along(apertura$paramSet)) {
  param_name <- apertura$paramSet[[i]]$id
  param_lower <- apertura$paramSet[[i]]$lower
  param_upper <- apertura$paramSet[[i]]$upper
  cat("  -", param_name, ": [", param_lower, ",", param_upper, "]\n")
}
cat("\n")

# Inicializar variables globales para trackear progreso
# Si ya existe el archivo log (proceso retomado), continuar desde ahí
if (file.exists(PARAMS$hyperparameter_tuning$files$output$BOlog)) {
  cat("Detectado log previo. Retomando desde iteración anterior...\n")
  tabla_log <- fread(PARAMS$hyperparameter_tuning$files$output$BOlog)
  GLOBAL_iteracion <- nrow(tabla_log)
  GLOBAL_ganancia <- tabla_log[, min(ganancia)]
  cat("Iteración inicial:", GLOBAL_iteracion, "\n")
  cat("Mejor ganancia hasta ahora:", GLOBAL_ganancia, "\n\n")
  rm(tabla_log)
} else {
  GLOBAL_iteracion <- 0
  GLOBAL_ganancia <- Inf
  cat("Comenzando optimización desde cero.\n\n")
}

# ----------------------------------------------------------------------------
# PASO 4: CONFIGURAR Y EJECUTAR OPTIMIZACIÓN BAYESIANA
# ----------------------------------------------------------------------------
cat("Paso 4: Configurando mlrMBO (framework de Optimización Bayesiana)...\n")

# Función a optimizar
funcion_optimizar <- EstimarGanancia_lightgbm

configureMlr(show.learner.output = FALSE)

# Configurar la función objetivo
# Esto le dice a BO:
# - Qué función llamar (EstimarGanancia_lightgbm)
# - Qué queremos (minimizar RMSE)
# - Qué hiperparámetros explorar (apertura$paramSet)
obj.fun <- makeSingleObjectiveFunction(
  fn = funcion_optimizar,
  minimize = PARAMS$hyperparameter_tuning$param$BO$minimize,
  noisy = PARAMS$hyperparameter_tuning$param$BO$noisy,
  par.set = makeParamSet(params = apertura$paramSet),
  has.simple.signature = PARAMS$hyperparameter_tuning$param$BO$has.simple.signature
)

# Configurar control de BO
# - Guardar progreso cada X segundos (por si se interrumpe)
# - Número de iteraciones a ejecutar
ctrl <- makeMBOControl(
  save.on.disk.at.time = PARAMS$hyperparameter_tuning$param$BO$save.on.disk.at.time,
  save.file.path = PARAMS$hyperparameter_tuning$files$output$BObin
)

ctrl <- setMBOControlTermination(
  ctrl,
  iters = PARAMS$hyperparameter_tuning$param$BO$iterations
)

ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI())

# Configurar el modelo sustituto (surrogate model)
# Este modelo aproxima la superficie de error en el espacio de hiperparámetros
# Usa Kriging con kernel Matérn 3/2 (buena opción para BO)
surr.km <- makeLearner(
  "regr.km",
  predict.type = "se",
  covtype = "matern3_2",
  control = list(trace = TRUE)
)

cat("Configuración completa.\n")
cat("Iteraciones a ejecutar:", PARAMS$hyperparameter_tuning$param$BO$iterations, "\n")
cat("\n=== INICIANDO OPTIMIZACIÓN BAYESIANA ===\n")
cat("Esto puede tomar varias horas...\n")
cat("El progreso se guarda en:", PARAMS$hyperparameter_tuning$files$output$BOlog, "\n\n")

# EJECUTAR OPTIMIZACIÓN BAYESIANA
if (!file.exists(PARAMS$hyperparameter_tuning$files$output$BObin)) {
  # Iniciar desde cero
  run <- mbo(obj.fun, learner = surr.km, control = ctrl)
} else {
  # Retomar desde archivo guardado (si se interrumpió)
  cat("Retomando optimización desde archivo guardado...\n")
  run <- mboContinue(PARAMS$hyperparameter_tuning$files$output$BObin)
}

cat("\n=== OPTIMIZACIÓN BAYESIANA COMPLETADA ===\n\n")

# ----------------------------------------------------------------------------
# PASO 5: ENTRENAR MODELO FINAL CON MEJORES HIPERPARÁMETROS
# ----------------------------------------------------------------------------
cat("Paso 5: Entrenando modelo final con mejores hiperparámetros...\n")

if (file.exists(PARAMS$hyperparameter_tuning$files$output$BOlog)) {

  # Cargar log y encontrar los mejores hiperparámetros
  tabla_log <- fread(PARAMS$hyperparameter_tuning$files$output$BOlog)
  mejor_iteracion <- tabla_log[which.min(ganancia)]

  cat("\n=== MEJORES HIPERPARÁMETROS ENCONTRADOS ===\n")
  cat("RMSE del mejor modelo:", mejor_iteracion$ganancia, "\n")
  cat("Iteración:", mejor_iteracion$iteracion_bayesiana, "\n")
  cat("Iteraciones de boosting:", mejor_iteracion$num_iterations, "\n\n")

  # Construir parámetros completos del mejor modelo
  mejores_params <- param_fijos

  # Agregar hiperparámetros optimizados
  for (col in names(apertura$paramSet)) {
    if (col %in% names(mejor_iteracion)) {
      mejores_params[[col]] <- mejor_iteracion[[col]]
      cat(col, "=", mejor_iteracion[[col]], "\n")
    }
  }

  mejores_params$num_iterations <- mejor_iteracion$num_iterations
  mejores_params$early_stopping_rounds <- NULL  # No usar early stopping en modelo final

  cat("\n")

  # ----------------------------------------------------------------------------
  # ENTRENAR CON TRAIN_FINAL (TODOS LOS DATOS HISTÓRICOS)
  # ----------------------------------------------------------------------------
  # Ahora que sabemos los mejores hiperparámetros, entrenamos un modelo
  # usando TODOS los datos disponibles (train + validate + test) para
  # maximizar el aprendizaje antes de predecir 2022.

  cat("Cargando train_final (todos los datos históricos)...\n")
  dataset_final <- fread(paste0("../02_TS/", PARAMS$training_strategy$files$output$train_final))

  cat("Train_final:", nrow(dataset_final), "filas x", ncol(dataset_final), "columnas\n")

  campos_buenos_final <- setdiff(copy(colnames(dataset_final)),
                                c(PARAMS$hyperparameter_tuning$const$campo_clase))

  dtrain_final <- lgb.Dataset(
    data = data.matrix(dataset_final[, campos_buenos_final, with = FALSE]),
    label = dataset_final[[PARAMS$hyperparameter_tuning$const$campo_clase]],
    free_raw_data = FALSE
  )

  cat("Entrenando modelo final...\n")
  set.seed(mejores_params$seed)
  modelo_final <- lgb.train(
    data = dtrain_final,
    param = mejores_params,
    verbose = 100  # Mostrar progreso cada 100 iteraciones
  )

  cat("\nModelo final entrenado exitosamente.\n")

  # ----------------------------------------------------------------------------
  # GUARDAR MODELO Y RESULTADOS
  # ----------------------------------------------------------------------------

  # Guardar modelo final (para uso posterior)
  saveRDS(modelo_final, file = "modelo_final_lgb.rds")
  cat("Modelo guardado como: modelo_final_lgb.rds\n")

  # Guardar importancia de variables
  # Esto indica qué variables fueron más útiles para las predicciones
  tb_importancia_final <- as.data.table(lgb.importance(modelo_final))
  fwrite(tb_importancia_final,
         file = PARAMS$hyperparameter_tuning$files$output$tb_importancia,
         sep = "\t")
  cat("Importancia de variables guardada como:",
      PARAMS$hyperparameter_tuning$files$output$tb_importancia, "\n")

  # Mostrar las 10 variables más importantes
  cat("\n=== TOP 10 VARIABLES MÁS IMPORTANTES ===\n")
  print(tb_importancia_final[1:min(10, nrow(tb_importancia_final))])
  cat("\n")

  # ----------------------------------------------------------------------------
  # PASO 6: GENERAR PREDICCIONES PARA EL AÑO 2022 (PRESENT DATA)
  # ----------------------------------------------------------------------------
  cat("Paso 6: Generando predicciones para el año 2022...\n")

  if (file.exists(paste0("../02_TS/", PARAMS$training_strategy$files$output$present_data))) {

    dataset_present <- fread(paste0("../02_TS/", PARAMS$training_strategy$files$output$present_data))

    if (nrow(dataset_present) > 0) {

      cat("Datos presentes (año 2021):", nrow(dataset_present), "registros\n")

      # Asegurar que usamos solo las variables que el modelo conoce
      campos_present <- intersect(campos_buenos_final, names(dataset_present))

      cat("Aplicando modelo...\n")
      predicciones_present <- predict(
        modelo_final,
        data.matrix(dataset_present[, campos_present, with = FALSE])
      )

      # Agregar predicciones al dataset
      dataset_present[, prediccion_clase := predicciones_present]

      # Guardar predicciones
      fwrite(dataset_present, file = "predicciones_presente.csv")

      cat("Predicciones guardadas como: predicciones_presente.csv\n")
      cat("\nEstadísticas de las predicciones:\n")
      cat("  - Media:", mean(predicciones_present), "\n")
      cat("  - Mediana:", median(predicciones_present), "\n")
      cat("  - Mínimo:", min(predicciones_present), "\n")
      cat("  - Máximo:", max(predicciones_present), "\n")
      cat("  - Desv. Est.:", sd(predicciones_present), "\n")

    } else {
      cat("No hay datos presentes para predecir.\n")
    }

  } else {
    cat("Archivo de datos presentes no encontrado.\n")
  }

  rm(tabla_log)
}

# ----------------------------------------------------------------------------
# RESUMEN FINAL
# ----------------------------------------------------------------------------
cat("\n=== HYPERPARAMETER TUNING COMPLETADO ===\n\n")

cat("ARCHIVOS GENERADOS:\n")
cat("  1.", PARAMS$hyperparameter_tuning$files$output$BOlog,
    "(log de todas las iteraciones - REVISAR)\n")
cat("  2. modelo_final_lgb.rds (modelo entrenado - LISTO PARA USAR)\n")
cat("  3.", PARAMS$hyperparameter_tuning$files$output$tb_importancia,
    "(importancia de variables)\n")
cat("  4. predicciones_presente.csv (predicciones para 2022)\n")
cat("  5. impo_*.txt (archivos de importancia intermedios)\n\n")

cat("PRÓXIMOS PASOS:\n")
cat("  1. Revisar el log de BO para entender qué hiperparámetros funcionaron mejor\n")
cat("  2. Analizar la importancia de variables para entender el modelo\n")
cat("  3. Revisar las predicciones para 2022 en predicciones_presente.csv\n")
cat("  4. Comparar las predicciones con los valores reales cuando estén disponibles\n")
cat("  5. Iterar: modificar feature engineering y volver a ejecutar\n\n")

cat("¡Felicitaciones! Has completado el pipeline completo de Machine Learning.\n")
cat("El modelo está listo para generar predicciones de gasto de bolsillo en salud.\n\n")

