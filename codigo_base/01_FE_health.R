# ============================================================================
# FEATURE ENGINEERING PARA HEALTH ECONOMICS
# Universidad del Oeste (UNO) - Aplicaciones en Ciencia de Datos
# ============================================================================
# Este script realiza la ingeniería de características (Feature Engineering)
# sobre el dataset de health economics para predecir el gasto de bolsillo
# en salud (hf3_ppp_pc) para el año 2022.
#
# El Feature Engineering es el proceso de crear nuevas variables predictoras
# a partir de las variables originales del dataset, para mejorar el desempeño
# del modelo de Machine Learning.
# ============================================================================

# ----------------------------------------------------------------------------
# LIBRERÍAS NECESARIAS
# ----------------------------------------------------------------------------
require("data.table")   # Manejo eficiente de datos
require("Rcpp")         # Para funciones en C++ (mayor velocidad)
require("rlist")        # Manejo de listas
require("yaml")         # Leer archivos de configuración YAML
library(dplyr)          # Manipulación de datos
library(stringr)        # Manejo de strings
library(lubridate)      # Manejo de fechas

require("lightgbm")      # Algoritmo de Machine Learning
require("randomForest")  # Para imputar valores faltantes

# ----------------------------------------------------------------------------
# FUNCIONES DE UTILIDAD
# ----------------------------------------------------------------------------
setwd("G:/Mi unidad/Especializacion/Aplicaciones de ciencia de datos/Tp final/health_economics_challenge")

# Reporta la cantidad de campos (columnas) del dataset
ReportarCampos <- function(dataset) {
  cat("La cantidad de campos es", ncol(dataset), "\n")
}

# ----------------------------------------------------------------------------
# AGREGAR VARIABLE CÍCLICA DE AÑO
# ----------------------------------------------------------------------------
AgregarMes <- function(dataset) {
  gc()  # Garbage collection para liberar memoria
  
  # Ciclo global por año (de 1 a 10, repetido)
  dataset[, year_cycle := ((year - min(year, na.rm = TRUE)) %% 10) + 1]
  
  ReportarCampos(dataset)
}

# ----------------------------------------------------------------------------
# ELIMINAR VARIABLES CON DATA DRIFTING
# ----------------------------------------------------------------------------
DriftEliminar <- function(dataset, variables) {
  gc()
  dataset[, c(variables) := NULL]
  ReportarCampos(dataset)
}

# ----------------------------------------------------------------------------
# CREAR DUMMIES PARA VALORES FALTANTES
# ----------------------------------------------------------------------------
DummiesNA <- function(dataset) {
  gc()
  
  # NAs en el año presente
  nulos <- colSums(is.na(dataset[year %in% PARAMS$feature_engineering$const$presente]))
  
  colsconNA <- names(which(nulos > 0))
  
  dataset[, paste0(colsconNA, "_isNA") := lapply(.SD, is.na),
          .SDcols = colsconNA]
  
  ReportarCampos(dataset)
}

# ----------------------------------------------------------------------------
# AGREGAR VARIABLES MANUALES (ACÁ VA TU PARTE)
# ----------------------------------------------------------------------------
AgregarVariables <- function(dataset) {
  gc()
  
  # ==========================
  # 1) YearsSinceFirst (ya lo tenías)
  # ==========================
  dataset[hf3_ppp_pc > 0, FirstYear := min(year, na.rm = TRUE),
          by = .(region, `Country Code`)]
  
  dataset[, FirstYear := nafill(FirstYear, type = "locf"),
          by = .(region, `Country Code`)]
  dataset[, FirstYear := nafill(FirstYear, type = "nocb"),
          by = .(region, `Country Code`)]
  
  dataset[, YearsSinceFirst := year - FirstYear]
  
  # ==========================
  # 2) Ratios y transformaciones de salud/economía
  #    controlado por: feature_engineering$param$health_ratios
  # ==========================
  if (isTRUE(PARAMS$feature_engineering$param$health_ratios)) {
    
    # Log del gasto de bolsillo (target original)
    if ("hf3_ppp_pc" %in% names(dataset)) {
      dataset[, hf3_ppp_pc_log := log1p(hf3_ppp_pc)]
    }
    
    # Log del PIB per cápita PPP
    if ("NY.GDP.PCAP.PP.CD" %in% names(dataset)) {
      dataset[, NY.GDP.PCAP.PP.CD_log := log1p(NY.GDP.PCAP.PP.CD)]
    }
    
    # Ratio: gasto de bolsillo / PIB per cápita PPP
    if (all(c("hf3_ppp_pc", "NY.GDP.PCAP.PP.CD") %in% names(dataset))) {
      dataset[, oop_gdp_ppp_ratio := hf3_ppp_pc / (NY.GDP.PCAP.PP.CD + 1)]
    }
    
    # Ratio: gasto de bolsillo / gasto en salud per cápita
    if (all(c("hf3_ppp_pc", "SH.XPD.CHEX.PC.CD") %in% names(dataset))) {
      dataset[, oop_health_pc_ratio := hf3_ppp_pc / (SH.XPD.CHEX.PC.CD + 1)]
    }
    
    # Ratio: gasto en salud per cápita / PIB per cápita PPP
    if (all(c("SH.XPD.CHEX.PC.CD", "NY.GDP.PCAP.PP.CD") %in% names(dataset))) {
      dataset[, healthpc_gdp_ppp_ratio := SH.XPD.CHEX.PC.CD / (NY.GDP.PCAP.PP.CD + 1)]
    }
    
    # Interacción: expectativa de vida * gasto en salud per cápita
    if (all(c("SP.DYN.LE00.IN", "SH.XPD.CHEX.PC.CD") %in% names(dataset))) {
      dataset[, lifeexp_x_healthpc := SP.DYN.LE00.IN * SH.XPD.CHEX.PC.CD]
    }
    
    # Interacción: % población >=65 * gasto de bolsillo
    if (all(c("SP.POP.65UP.TO.ZS", "hf3_ppp_pc") %in% names(dataset))) {
      dataset[, pop65_x_oop := SP.POP.65UP.TO.ZS * hf3_ppp_pc]
    }
    
    # Interacción: % urbana * gasto de bolsillo
    if (all(c("SP.URB.TOTL.IN.ZS", "hf3_ppp_pc") %in% names(dataset))) {
      dataset[, urban_x_oop := SP.URB.TOTL.IN.ZS * hf3_ppp_pc]
    }
  }
  
  # ==========================
  # 3) Proxy de QALYs
  #    controlado por: feature_engineering$param$qaly_calculation
  # ==========================
  if (isTRUE(PARAMS$feature_engineering$param$qaly_calculation)) {
    
    # Necesitamos expectativa de vida y mortalidad infantil
    if (all(c("SP.DYN.LE00.IN", "SP.DYN.IMRT.IN") %in% names(dataset))) {
      # Mortalidad infantil está en muertes por 1.000 nacidos vivos.
      # Suponemos que mayor mortalidad => peor calidad.
      dataset[, qaly_proxy := SP.DYN.LE00.IN * (1 - SP.DYN.IMRT.IN / 1000)]
      # Evitar valores negativos raros
      dataset[qaly_proxy < 0, qaly_proxy := NA_real_]
    }
  }
  
  # ==========================
  # 4) Dummies de crisis (2008) y COVID (2020–2021)
  #    controlado por: feature_engineering$param$crisis_dummies
  # ==========================
  if (isTRUE(PARAMS$feature_engineering$param$crisis_dummies)) {
    
    if ("year" %in% names(dataset)) {
      # Crisis 2008–2009
      dataset[, dummy_crisis_2008 := as.integer(year %in% c(2008, 2009))]
      
      # COVID: 2020–2021
      dataset[, dummy_covid_2020 := as.integer(year %in% c(2020, 2021))]
    }
  }
  
  # ==========================
  # 5) Válvulas de seguridad (como antes)
  # ==========================
  
  # Infinitos -> NA
  infinitos <- lapply(names(dataset), function(.name) dataset[, sum(is.infinite(get(.name)))])
  infinitos_qty <- sum(unlist(infinitos))
  
  if (infinitos_qty > 0) {
    cat("ATENCIÓN: hay", infinitos_qty, "valores infinitos en tu dataset. Serán pasados a NA\n")
    dataset[mapply(is.infinite, dataset)] <<- NA
  }
  
  # NaN -> 0
  nans <- lapply(names(dataset), function(.name) dataset[, sum(is.nan(get(.name)))])
  nans_qty <- sum(unlist(nans))
  
  if (nans_qty > 0) {
    cat("ATENCIÓN: hay", nans_qty, "valores NaN (0/0) en tu dataset. Serán pasados arbitrariamente a 0\n")
    cat("Si no te gusta la decisión, modifica a gusto el programa!\n\n")
    dataset[mapply(is.nan, dataset)] <<- 0
  }
  
  ReportarCampos(dataset)
}

# ----------------------------------------------------------------------------
# CALCULAR LAGS (VALORES RETRASADOS)
# ----------------------------------------------------------------------------
Lags <- function(cols, nlag, deltas) {
  gc()
  sufijo <- paste0("_lag", nlag)
  
  dataset[, paste0(cols, sufijo) := shift(.SD, nlag, NA, "lag"),
          by = `Country Code`,
          .SDcols = cols]
  
  if (deltas) {
    sufijodelta <- paste0("_delta", nlag)
    
    for (vcol in cols) {
      dataset[, paste0(vcol, sufijodelta) := get(vcol) - get(paste0(vcol, sufijo))]
    }
  }
  
  ReportarCampos(dataset)
}

# ----------------------------------------------------------------------------
# FUNCIÓN C++ PARA ESTADÍSTICAS DE VENTANA
# ----------------------------------------------------------------------------
cppFunction('NumericVector fhistC(NumericVector pcolumna, IntegerVector pdesde)
{
  double  x[100];
  double  y[100];

  int n = pcolumna.size();
  NumericVector out(5*n);

  for(int i = 0; i < n; i++)
  {
    if(pdesde[i]-1 < i)  out[i + 4*n] = pcolumna[i-1];
    else                 out[i + 4*n] = NA_REAL;

    int libre = 0;
    int xvalor = 1;

    for(int j = pdesde[i]-1; j <= i; j++)
    {
      double a = pcolumna[j];

      if(!R_IsNA(a))
      {
        y[libre] = a;
        x[libre] = xvalor;
        libre++;
      }

      xvalor++;
    }

    if(libre > 1)
    {
      double xsum  = x[0];
      double ysum  = y[0];
      double xysum = xsum * ysum;
      double xxsum = xsum * xsum;
      double vmin  = y[0];
      double vmax  = y[0];

      for(int h = 1; h < libre; h++)
      {
        xsum  += x[h];
        ysum  += y[h];
        xysum += x[h] * y[h];
        xxsum += x[h] * x[h];

        if(y[h] < vmin) vmin = y[h];
        if(y[h] > vmax) vmax = y[h];
      }

      out[i]        = (libre*xysum - xsum*ysum) / (libre*xxsum - xsum*xsum);
      out[i + n]    = vmin;
      out[i + 2*n]  = vmax;
      out[i + 3*n]  = ysum / libre;
    }
    else
    {
      out[i]       = NA_REAL;
      out[i + n]   = NA_REAL;
      out[i + 2*n] = NA_REAL;
      out[i + 3*n] = NA_REAL;
    }
  }

  return out;
}')

# ----------------------------------------------------------------------------
# TENDENCIAS Y ESTADÍSTICAS MÓVILES
# ----------------------------------------------------------------------------
TendenciaYmuchomas <- function(dataset, cols, ventana = 6, tendencia = TRUE,
                               minimo = TRUE, maximo = TRUE, promedio = TRUE,
                               ratioavg = FALSE, ratiomax = FALSE) {
  gc()
  
  ventana_regresion <- ventana
  last <- nrow(dataset)
  
  vector_ids <- dataset$`Country Code`
  
  vector_desde <- seq(-ventana_regresion + 2, nrow(dataset) - ventana_regresion + 1)
  vector_desde[1:ventana_regresion] <- 1
  
  for (i in 2:last) {
    if (vector_ids[i - 1] != vector_ids[i]) {
      vector_desde[i] <- i
    }
  }
  
  for (i in 2:last) {
    if (vector_desde[i] < vector_desde[i - 1]) {
      vector_desde[i] <- vector_desde[i - 1]
    }
  }
  
  for (campo in cols) {
    nueva_col <- fhistC(dataset[, get(campo)], vector_desde)
    
    if (tendencia) {
      dataset[, paste0(campo, "_tend", ventana) := nueva_col[(0*last + 1):(1*last)]
      ]
    }
    if (minimo) {
      dataset[, paste0(campo, "_min", ventana)  := nueva_col[(1*last + 1):(2*last)]
      ]
    }
    if (maximo) {
      dataset[, paste0(campo, "_max", ventana)  := nueva_col[(2*last + 1):(3*last)]
      ]
    }
    if (promedio) {
      dataset[, paste0(campo, "_avg", ventana)  := nueva_col[(3*last + 1):(4*last)]
      ]
    }
    if (ratioavg) {
      dataset[, paste0(campo, "_ratioavg", ventana) :=
                get(campo) / nueva_col[(3*last + 1):(4*last)]]
    }
    if (ratiomax) {
      dataset[, paste0(campo, "_ratiomax", ventana) :=
                get(campo) / nueva_col[(2*last + 1):(3*last)]]
    }
  }
  
  ReportarCampos(dataset)
}

# ----------------------------------------------------------------------------
# SELECCIÓN DE VARIABLES POR IMPORTANCIA (CANARITOS)
# ----------------------------------------------------------------------------
GVEZ <- 1

CanaritosImportancia <- function(canaritos_ratio = 0.2) {
  gc()
  ReportarCampos(dataset)
  
  canaritos_year_end <- PARAMS$feature_engineering$const$canaritos_year_end
  
  for (i in 1:(ncol(dataset) * canaritos_ratio)) {
    dataset[, paste0("canarito", i) := runif(nrow(dataset))]
  }
  
  campos_buenos <- setdiff(colnames(dataset),
                           c("Country Code", "year", PARAMS$feature_engineering$const$clase))
  
  azar <- runif(nrow(dataset))
  dataset[, entrenamiento :=
            year >= PARAMS$feature_engineering$const$canaritos_year_start &
            year < canaritos_year_end &
            azar < 0.10]
  
  dtrain <- lgb.Dataset(
    data = data.matrix(dataset[entrenamiento == TRUE, campos_buenos, with = FALSE]),
    label = dataset[entrenamiento == TRUE, get(PARAMS$feature_engineering$const$clase)],
    free_raw_data = FALSE
  )
  
  canaritos_year_valid <- PARAMS$feature_engineering$const$canaritos_year_valid
  
  dvalid <- lgb.Dataset(
    data = data.matrix(dataset[year == canaritos_year_valid, campos_buenos, with = FALSE]),
    label = dataset[year == canaritos_year_valid, get(PARAMS$feature_engineering$const$clase)],
    free_raw_data = FALSE
  )
  
  param <- list(
    objective = "regression",
    metric = "rmse",
    first_metric_only = TRUE,
    boost_from_average = TRUE,
    feature_pre_filter = FALSE,
    verbosity = -100,
    seed = 999983,
    max_depth = -1,
    min_gain_to_split = 0.0,
    lambda_l1 = 0.0,
    lambda_l2 = 0.0,
    max_bin = 1023,
    num_iterations = 500,
    force_row_wise = TRUE,
    learning_rate = 0.065,
    feature_fraction = 1.0,
    min_data_in_leaf = 260,
    num_leaves = 60,
    early_stopping_rounds = 50
  )
  
  modelo <- lgb.train(
    data = dtrain,
    valids = list(valid = dvalid),
    param = param,
    verbose = -100
  )
  
  tb_importancia <- lgb.importance(model = modelo)
  tb_importancia[, pos := .I]
  
  GVEZ <<- GVEZ + 1
  
  umbral <- tb_importancia[Feature %like% "canarito", median(pos) + 2*sd(pos)]
  
  col_utiles <- tb_importancia[pos < umbral & !(Feature %like% "canarito"), Feature]
  
  col_utiles <- unique(c(col_utiles,
                         c("Country Code", "year",
                           PARAMS$feature_engineering$const$clase, "year_cycle")))
  
  col_inutiles <- setdiff(colnames(dataset), col_utiles)
  dataset[, (col_inutiles) := NULL]
  
  ReportarCampos(dataset)
}

# ----------------------------------------------------------------------------
# CREAR RANKINGS RELATIVOS
# ----------------------------------------------------------------------------
Rankeador <- function(cols) {
  gc()
  sufijo <- "_rank"
  
  for (vcol in cols) {
    dataset[, paste0(vcol, sufijo) :=
              frank(get(vcol), ties.method = "random") / .N,
            by = year]
  }
  
  ReportarCampos(dataset)
}

# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================

# Cargar el dataset original
setwd(PARAMS$environment$base_dir)
setwd(PARAMS$environment$data_dir)
nom_arch <- PARAMS$feature_engineering$files$input$dentrada
dataset <- fread(nom_arch)

# Ordenar por Country Code y year
setorderv(dataset, PARAMS$feature_engineering$const$campos_sort)

# ----------------------------------------------------------------------------
# CREAR VARIABLE OBJETIVO (CLASE) CON LEAD
# ----------------------------------------------------------------------------
dataset[, PARAMS$feature_engineering$const$clase :=
          get(PARAMS$feature_engineering$const$origen_clase),
        by = c("region", "Country Code")]

dataset[, PARAMS$feature_engineering$const$clase :=
          shift(get(PARAMS$feature_engineering$const$clase),
                n = PARAMS$feature_engineering$const$orden_lead,
                type = "lead"),
        by = c("region", "Country Code")]

# ----------------------------------------------------------------------------
# APLICAR TRANSFORMACIONES SEGÚN YAML
# ----------------------------------------------------------------------------
AgregarMes(dataset)

if (length(PARAMS$feature_engineering$param$variablesdrift) > 0) {
  DriftEliminar(dataset, PARAMS$feature_engineering$param$variablesdrift)
}

if (PARAMS$feature_engineering$param$dummiesNA) {
  DummiesNA(dataset)
}

if (PARAMS$feature_engineering$param$variablesmanuales) {
  AgregarVariables(dataset)
}

dataset[, PARAMS$feature_engineering$const$origen_clase := NULL]

# ----------------------------------------------------------------------------
# CALCULAR LAGS Y TENDENCIAS
# ----------------------------------------------------------------------------
cols_lagueables <- copy(setdiff(colnames(dataset),
                                PARAMS$feature_engineering$const$campos_fijos))

for (i in 1:length(PARAMS$feature_engineering$param$tendenciaYmuchomas$correr)) {
  if (PARAMS$feature_engineering$param$tendenciaYmuchomas$correr[i]) {
    
    if (PARAMS$feature_engineering$param$acumulavars) {
      cols_lagueables <- setdiff(colnames(dataset),
                                 PARAMS$feature_engineering$const$campos_fijos)
    }
    
    cols_lagueables <- intersect(colnames(dataset), cols_lagueables)
    
    TendenciaYmuchomas(
      dataset,
      cols      = cols_lagueables,
      ventana   = PARAMS$feature_engineering$param$tendenciaYmuchomas$ventana[i],
      tendencia = PARAMS$feature_engineering$param$tendenciaYmuchomas$tendencia[i],
      minimo    = PARAMS$feature_engineering$param$tendenciaYmuchomas$minimo[i],
      maximo    = PARAMS$feature_engineering$param$tendenciaYmuchomas$maximo[i],
      promedio  = PARAMS$feature_engineering$param$tendenciaYmuchomas$promedio[i],
      ratioavg  = PARAMS$feature_engineering$param$tendenciaYmuchomas$ratioavg[i],
      ratiomax  = PARAMS$feature_engineering$param$tendenciaYmuchomas$ratiomax[i]
    )
    
    if (PARAMS$feature_engineering$param$tendenciaYmuchomas$canaritos[i] > 0) {
      CanaritosImportancia(
        canaritos_ratio = unlist(PARAMS$feature_engineering$param$tendenciaYmuchomas$canaritos[i])
      )
    }
  }
}

for (i in 1:length(PARAMS$feature_engineering$param$lags$correr)) {
  if (PARAMS$feature_engineering$param$lags$correr[i]) {
    
    if (PARAMS$feature_engineering$param$acumulavars) {
      cols_lagueables <- setdiff(colnames(dataset),
                                 PARAMS$feature_engineering$const$campos_fijos)
    }
    
    cols_lagueables <- intersect(colnames(dataset), cols_lagueables)
    
    Lags(
      cols_lagueables,
      PARAMS$feature_engineering$param$lags$lag[i],
      PARAMS$feature_engineering$param$lags$delta[i]
    )
    
    if (PARAMS$feature_engineering$param$lags$canaritos[i] > 0) {
      CanaritosImportancia(
        canaritos_ratio = unlist(PARAMS$feature_engineering$param$lags$canaritos[i])
      )
    }
  }
}

if (PARAMS$feature_engineering$param$acumulavars) {
  cols_lagueables <- setdiff(colnames(dataset),
                             PARAMS$feature_engineering$const$campos_fijos)
}

# ----------------------------------------------------------------------------
# CREAR RANKINGS
# ----------------------------------------------------------------------------
if (PARAMS$feature_engineering$param$rankeador) {
  
  if (PARAMS$feature_engineering$param$acumulavars) {
    cols_lagueables <- setdiff(colnames(dataset),
                               PARAMS$feature_engineering$const$campos_fijos)
  }
  
  cols_lagueables <- intersect(colnames(dataset), cols_lagueables)
  
  setorderv(dataset, PARAMS$feature_engineering$const$campos_rsort)
  Rankeador(cols_lagueables)
  setorderv(dataset, PARAMS$feature_engineering$const$campos_sort)
}

if (PARAMS$feature_engineering$param$canaritos_final > 0) {
  CanaritosImportancia(canaritos_ratio = PARAMS$feature_engineering$param$canaritos_final)
}

# ----------------------------------------------------------------------------
# LIMPIEZA FINAL
# ----------------------------------------------------------------------------
nuevo_orden <- c(setdiff(colnames(dataset), PARAMS$feature_engineering$const$clase),
                 PARAMS$feature_engineering$const$clase)
setcolorder(dataset, nuevo_orden)

cols_lagueables <- copy(setdiff(colnames(dataset),
                                PARAMS$feature_engineering$const$campos_fijos))
for (col in cols_lagueables) {
  dataset[[col]][is.nan(dataset[[col]])] <- 0
}

dataset <- dataset[(!is.na(get(PARAMS$feature_engineering$const$clase))) |
                     year >= PARAMS$feature_engineering$const$presente]

# ----------------------------------------------------------------------------
# GUARDAR DATASET PROCESADO
# ----------------------------------------------------------------------------
experiment_dir <- paste(PARAMS$experiment$experiment_label,
                        PARAMS$experiment$experiment_code, sep = "_")
experiment_lead_dir <- paste(PARAMS$experiment$experiment_label,
                             PARAMS$experiment$experiment_code,
                             paste0("f", PARAMS$feature_engineering$const$orden_lead),
                             sep = "_")

setwd(PARAMS$environment$base_dir)
setwd(paste0(PARAMS$experiment$exp_dir))

dir.create(experiment_dir, showWarnings = FALSE)
setwd(experiment_dir)
dir.create(experiment_lead_dir, showWarnings = FALSE)
setwd(experiment_lead_dir)

PARAMS$features$features_n <- length(colnames(dataset))
PARAMS$features$colnames   <- colnames(dataset)

jsontest <- jsonlite::toJSON(PARAMS, pretty = TRUE, auto_unbox = TRUE)
sink(file = paste0(experiment_lead_dir, ".json"))
print(jsontest)
sink(file = NULL)

dir.create("01_FE", showWarnings = FALSE)
setwd("01_FE")

fwrite(dataset,
       paste0(experiment_lead_dir, ".csv.gz"),
       logical01 = TRUE,
       sep = ",")

cat("\n=== FEATURE ENGINEERING HEALTH ECONOMICS COMPLETADO ===\n")
cat("Dimensiones finales del dataset:", nrow(dataset), "x", ncol(dataset), "\n")
cat("Archivo guardado exitosamente\n")
cat("\nPróximos pasos:\n")
cat("1. Revisar el archivo JSON generado para ver qué variables se crearon\n")
cat("2. Ejecutar el script 02_TS_health.R para particionar los datos\n")
cat("3. Experimentar creando tus propias variables en AgregarVariables()\n")
cat("\n¡Buena suerte con el desafío!\n")

