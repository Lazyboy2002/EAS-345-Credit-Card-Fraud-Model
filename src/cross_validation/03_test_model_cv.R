library(data.table)
library(xgboost)
library(pROC)

test_set <- readRDS("data/processed/test_set.rds")
final_model <- xgb.load("models/final_xgb_model_cv.model")

test_labels <- test_set$isFraud

test_set$datetime_numeric <- as.numeric(test_set$datetime)

features <- setdiff(names(test_set), c("isFraud", "datetime"))
test_matrix <- data.matrix(test_set[, ..features])

dtest <- xgb.DMatrix(data = test_matrix, label = test_labels)

test_pred <- predict(final_model, dtest)
test_pred_class <- ifelse(test_pred > 0.5, 1, 0)

confusion <- table(Predicted = test_pred_class, Actual = test_labels)
print(confusion)

# Compute precision, recall, F1
precision <- sum(test_pred_class == 1 & test_labels == 1) / sum(test_pred_class == 1)
recall    <- sum(test_pred_class == 1 & test_labels == 1) / sum(test_labels == 1)
f1_score  <- 2 * precision * recall / (precision + recall)

cat(sprintf("Precision: %.4f, Recall: %.4f, F1: %.4f\n", precision, recall, f1_score))


importance_matrix <- xgb.importance(feature_names = features, model = final_model)
print(importance_matrix)

# Optional: plot importance
xgb.plot.importance(importance_matrix, top_n = 20)

roc_obj <- roc(test_labels, test_pred)
auc_value <- auc(roc_obj)
cat(sprintf("Test AUC: %.4f\n", auc_value))