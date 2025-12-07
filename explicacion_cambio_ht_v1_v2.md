# Explicación del Cambio: Hyperparameter Tuning v1 → v2

**Universidad Nacional del Oeste (UNO)**  
**Curso:** Aplicaciones en Ciencia de Datos  
**Tema:** Corrección del data leakage en optimización bayesiana

---

## 🎯 El Problema

En la versión 1 del código, cometíamos un error sutil pero importante: usábamos el **mismo conjunto de datos (VALIDATE)** para dos propósitos diferentes:

1. **Early stopping**: decidir cuándo parar de entrenar
2. **Evaluar hiperparámetros**: reportar el RMSE a la optimización bayesiana

Esto genera un sesgo optimista en la evaluación de hiperparámetros.

---

## 📝 Código v1 (PROBLEMÁTICO)

```r
EstimarGanancia_lightgbm <- function(x) {
  
  # Hiperparámetros a optimizar
  param_completo <- list(
    boosting = "gbdt",
    objective = "regression",
    metric = "rmse",
    learning_rate = x$learning_rate,
    num_leaves = x$num_leaves,
    min_data_in_leaf = x$min_data_in_leaf,
    feature_fraction = x$feature_fraction,
    bagging_fraction = x$bagging_fraction,
    bagging_freq = x$bagging_freq,
    lambda_l1 = x$lambda_l1,
    lambda_l2 = x$lambda_l2,
    max_depth = x$max_depth,
    verbose = -1
  )
  
  # Entrenar modelo
  modelo <- lgb.train(
    data = dtrain,
    valids = list(valid = dvalidate),
    params = param_completo,
    nrounds = 2000,
    early_stopping_rounds = 50,
    verbose = -1
  )
  
  # ⚠️ PROBLEMA: Tomamos el RMSE del mismo conjunto usado para early stopping
  best_iter <- modelo$best_iter
  rmse_validate <- modelo$record_evals$valid$rmse[[best_iter]]
  
  return(list(Score = -rmse_validate, Pred = 0))
}
```

### ¿Por qué es problemático?

| Paso | Qué usa | Conjunto |
|------|---------|----------|
| Early stopping | `valids = list(valid = dvalidate)` | VALIDATE (2019) |
| RMSE para BO | `modelo$record_evals$valid$rmse` | VALIDATE (2019) ⚠️ |

El modelo ya "vio" VALIDATE durante el entrenamiento para decidir cuándo parar. Después usamos **ese mismo RMSE** para guiar la optimización bayesiana. Es como si el árbitro del entrenamiento también fuera el juez de la competencia final.

---

## ✅ Código v2 (CORRECTO)

```r
EstimarGanancia_lightgbm <- function(x) {
  
  # Hiperparámetros a optimizar
  param_completo <- list(
    boosting = "gbdt",
    objective = "regression",
    metric = "rmse",
    learning_rate = x$learning_rate,
    num_leaves = x$num_leaves,
    min_data_in_leaf = x$min_data_in_leaf,
    feature_fraction = x$feature_fraction,
    bagging_fraction = x$bagging_fraction,
    bagging_freq = x$bagging_freq,
    lambda_l1 = x$lambda_l1,
    lambda_l2 = x$lambda_l2,
    max_depth = x$max_depth,
    verbose = -1
  )
  
  # Entrenar modelo (early stopping usa VALIDATE)
  modelo <- lgb.train(
    data = dtrain,
    valids = list(valid = dvalidate),
    params = param_completo,
    nrounds = 2000,
    early_stopping_rounds = 50,
    verbose = -1
  )
  
  # ✅ CORRECTO: Evaluamos en TEST, un conjunto completamente independiente
  predicciones <- predict(modelo, dtest_matrix)
  rmse_test <- sqrt(mean((predicciones - datos_test$target)^2))
  
  return(list(Score = -rmse_test, Pred = 0))
}
```

### ¿Por qué es correcto?

| Paso | Qué usa | Conjunto |
|------|---------|----------|
| Early stopping | `valids = list(valid = dvalidate)` | VALIDATE (2019) |
| RMSE para BO | `predict(modelo, dtest_matrix)` | TEST (2020) ✅ |

Ahora cada conjunto tiene **un único propósito**:
- **VALIDATE** → Solo para early stopping (cuándo parar de agregar árboles)
- **TEST** → Evaluación "fresca" que guía la optimización bayesiana

---

## 🔍 Diferencia Lado a Lado

```r
# ═══════════════════════════════════════════════════════════════════════════
# v1: PROBLEMÁTICO
# ═══════════════════════════════════════════════════════════════════════════

  best_iter <- modelo$best_iter
  rmse_validate <- modelo$record_evals$valid$rmse[[best_iter]]
  #                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  #                       Usa el RMSE que el modelo ya vio
  #                       durante el proceso de early stopping
  
  return(list(Score = -rmse_validate, Pred = 0))


# ═══════════════════════════════════════════════════════════════════════════
# v2: CORRECTO
# ═══════════════════════════════════════════════════════════════════════════

  predicciones <- predict(modelo, dtest_matrix)
  #                               ^^^^^^^^^^^^
  #                               Datos que el modelo NUNCA vio
  #                               durante ninguna parte del entrenamiento
  
  rmse_test <- sqrt(mean((predicciones - datos_test$target)^2))
  
  return(list(Score = -rmse_test, Pred = 0))
```

---

## 📊 Impacto Esperado

| Aspecto | v1 | v2 |
|---------|----|----|
| RMSE reportado durante BO | Optimista (sesgado) | Realista |
| Riesgo de overfitting a hiperparámetros | Alto | Bajo |
| Generalización a datos futuros | Peor | Mejor |
| Honestidad de la evaluación | ❌ | ✅ |

---

## 🧠 Concepto Clave: Early Stopping

**Early stopping** es una técnica de regularización que detiene el entrenamiento cuando el modelo deja de mejorar en un conjunto de validación.

```
Iteración   RMSE (train)   RMSE (validate)
─────────   ────────────   ───────────────
   100         0.45            0.52        ← mejorando
   200         0.35            0.48        ← mejorando  
   300         0.25            0.45        ← MEJOR PUNTO ★
   400         0.18            0.46        ← validate empeora...
   450         0.15            0.47        ← sigue empeorando...
   500 ⛔      0.12            0.48        ← STOP! (50 rounds sin mejorar)
```

El modelo se queda con los parámetros de la **iteración 300** (mejor en validate).

### ¿Por qué necesitamos early stopping?

Sin early stopping, el modelo seguiría entrenando hasta las 2000 rondas, memorizando cada vez más el conjunto de entrenamiento y perdiendo capacidad de generalización.

---

## 📁 Archivos Relacionados

- `comparacion_ht_v1_v2.dot` → Diagrama visual del cambio
- `comparacion_ht_v1_v2.pdf` → Versión compilada del diagrama

---

## 🎓 Moraleja Pedagógica

> **"Cada conjunto de datos debe tener UN único propósito."**

En Machine Learning riguroso:
- **TRAIN** → Entrenar el modelo
- **VALIDATE** → Decisiones durante el entrenamiento (early stopping, selección de arquitectura)
- **TEST** → Evaluación final, completamente independiente

Cuando un conjunto hace "doble trabajo", perdemos la honestidad de nuestra evaluación.
