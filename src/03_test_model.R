# ============================================
# 03_test_model.R
# Test XGBoost model on preprocessed test set
# ============================================

# 1️ Load libraries
library(xgboost)
library(caret)
library(pROC)
library(PRROC)

# 2️ Load preprocessed test set
test_set <- readRDS("data/test_set.rds")
test_set   <- as.data.frame(test_set)

# 3 Separate features and target
X_test <- test_set[, !names(test_set) %in% "isFraud"]
y_test <- test_set$isFraud

# 4️ Convert all columns to numeric (handle POSIXct/logical if any)
X_test[] <- lapply(X_test, function(col) {
  if (inherits(col, "POSIXct") || inherits(col, "POSIXt")) as.numeric(col)
  else as.numeric(col)
})

# 5️ Convert to xgb.DMatrix
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)

# 6️ Load the trained XGBoost model
xgb_model <- xgb.load("output/xgb_fraud.model")

# 7️ Make predictions
pred_prob  <- predict(xgb_model, dtest)
pred_label <- ifelse(pred_prob > 0.5, 1, 0)

# 8️ Confusion Matrix
cm <- confusionMatrix(as.factor(pred_label), as.factor(y_test), positive = "1")
print(cm)

# 9️ ROC and AUC
roc_curve <- roc(y_test, pred_prob)
auc_val <- auc(roc_curve)
cat("AUC:", auc_val, "\n")
plot(roc_curve, main = "ROC Curve for XGBoost Fraud Model")

# 10️ Precision-Recall curve (for imbalanced datasets)
pr <- pr.curve(
  scores.class0 = pred_prob[y_test == 1],
  scores.class1 = pred_prob[y_test == 0],
  curve = TRUE
)
plot(pr, main = "Precision-Recall Curve")

# 11️ Optional: Top feature importance from trained model
importance_matrix <- xgb.importance(model = xgb_model)
print(importance_matrix[1:10, ])
xgb.plot.importance(importance_matrix)
