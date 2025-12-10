library(xgboost)

train_set <- readRDS("data/processed/train_set.rds")
val_set <- readRDS("data/processed/val_set.rds")
test_set <- readRDS("data/processed/test_set.rds")


# -----------------------------
# Step 0: Convert target
# -----------------------------
train_labels <- train_set$isFraud
val_labels   <- val_set$isFraud
test_labels  <- test_set$isFraud

# -----------------------------
# Step 1: Convert datetime to numeric
# -----------------------------
train_set$datetime_numeric <- as.numeric(train_set$datetime)
val_set$datetime_numeric   <- as.numeric(val_set$datetime)
test_set$datetime_numeric  <- as.numeric(test_set$datetime)

# -----------------------------
# Step 2: Select features (exclude target and original datetime)
# -----------------------------
features <- setdiff(names(train_set), c("isFraud", "datetime"))

# Use .. prefix for data.table variable
train_matrix <- data.matrix(train_set[, ..features])
val_matrix   <- data.matrix(val_set[, ..features])
test_matrix  <- data.matrix(test_set[, ..features])

# -----------------------------
# Step 3: Create DMatrix objects
# -----------------------------
dtrain <- xgb.DMatrix(data = train_matrix, label = train_labels)
dval   <- xgb.DMatrix(data = val_matrix, label = val_labels)
dtest  <- xgb.DMatrix(data = test_matrix, label = test_labels)

# -----------------------------
# Step 4: Set XGBoost parameters
# -----------------------------
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "aucpr",               # Precision-Recall AUC for imbalanced data
  eta = 0.05,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8,
  scale_pos_weight = sum(train_labels == 0) / sum(train_labels == 1) # handle class imbalance
)

# -----------------------------
# Step 5: k-Fold Cross-Validation
# -----------------------------
k <- 5
set.seed(123)

# Create folds manually
cv_folds <- split(seq_len(nrow(train_matrix)), sample(rep(1:k, length.out = nrow(train_matrix))))

cv_model <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 2000,
  folds = cv_folds,
  early_stopping_rounds = 50,
  maximize = TRUE,
  verbose = 1
)

best_nrounds <- cv_model$best_iteration
cat("Best number of boosting rounds:", best_nrounds, "\n")

# -----------------------------
# Step 6: Train final model using best nrounds
# -----------------------------
final_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,
  watchlist = list(train = dtrain, val = dval),
  early_stopping_rounds = 50,
  maximize = TRUE,
  verbose = 1
)

# -----------------------------
# Step 7: Evaluate on test set
# -----------------------------
test_pred <- predict(final_model, dtest)
test_pred_class <- ifelse(test_pred > 0.5, 1, 0)

confusion <- table(Predicted = test_pred_class, Actual = test_labels)
print(confusion)

# Optional: compute precision, recall, F1
precision <- sum(test_pred_class == 1 & test_labels == 1) / sum(test_pred_class == 1)
recall    <- sum(test_pred_class == 1 & test_labels == 1) / sum(test_labels == 1)
f1_score  <- 2 * precision * recall / (precision + recall)

cat(sprintf("Precision: %.4f, Recall: %.4f, F1: %.4f\n", precision, recall, f1_score))

# -----------------------------
# Step 8: Save model
# -----------------------------
xgb.save(final_model, "models/xgb_model_cv.model")
