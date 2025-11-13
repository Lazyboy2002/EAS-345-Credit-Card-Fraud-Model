# ============================================
# 03_test_model.R
# Test XGBoost model on preprocessed test set
# ============================================

# 1️ Load libraries
library(xgboost)
library(caret)
library(pROC)
library(PRROC)

# Load preprocessed test set
test_set <- readRDS("data/test_set.rds")

#ensure test set is a flat data fram
test_set   <- as.data.frame(test_set)

# Get a table that contains all columns except the target column isFraud
X_test <- test_set[, !names(test_set) %in% "isFraud"]

#extract the isFraud column from test set
y_test <- test_set$isFraud

# Convert all columns to numeric (handle POSIXct/logical if any) in test set
X_test[] <- lapply(X_test, function(col) {
  if (inherits(col, "POSIXct") || inherits(col, "POSIXt")) as.numeric(col)
  else as.numeric(col)
})

# Convert test set to xgb.DMatrix
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)

# Load the trained XGBoost model
xgb_model <- xgb.load("output/xgb_fraud.model")

# Make predictions on the test set that does not have the isFraud feature
#predictions are probabilities of fraud for every row in the test set
pred_prob  <- predict(xgb_model, dtest)

#convert probabilities into binary class labels using threshold of 0.5
#if predicted prob greater than 0.5, classify as 1 (fraud)
#else classify as 0 (not fraud)
pred_label <- ifelse(pred_prob > 0.5, 1, 0)

# create Confusion Matrix using caret package
#pred_label is predicted outcomes
#y_test is true outcomes
#positive is value of 1
cm <- confusionMatrix(as.factor(pred_label), as.factor(y_test), positive = "1")
print(cm)

# ROC and AUC

#receiver operating characteristic curve. 
#tradeoff between true positive rate and false positive rate
#how well model separates fraud from non fraud across probability cutoffs
roc_curve <- roc(y_test, pred_prob)

#area under the ROC curve
#measures how well the model distinguished between positive and negative class
auc_val <- auc(roc_curve)

cat("AUC:", auc_val, "\n")
plot(roc_curve, main = "ROC Curve for XGBoost Fraud Model")

# Precision-Recall curve (for imbalanced datasets)
#scores.class0 - predicted probabilities for the positive class, fraud rows
#scores.class1 - predicted probabilities for the negative class, not fraud rows
pr <- pr.curve(
  scores.class0 = pred_prob[y_test == 1],
  scores.class1 = pred_prob[y_test == 0],
  curve = TRUE
)
plot(pr, main = "Precision-Recall Curve")

#print feature importance matrix, which features were most useful for making predictions
importance_matrix <- xgb.importance(model = xgb_model)

#print top 10 features
print(importance_matrix[1:10, ])

#plot importance matrix
xgb.plot.importance(importance_matrix)
