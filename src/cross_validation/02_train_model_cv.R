library(data.table)
library(xgboost)

train_set <- readRDS("data/processed/train_set.rds")
val_set   <- readRDS("data/processed/val_set.rds")

train_labels <- train_set$isFraud
val_labels   <- val_set$isFraud

train_set$datetime_numeric <- as.numeric(train_set$datetime)
val_set$datetime_numeric   <- as.numeric(val_set$datetime)

features <- setdiff(names(train_set), c("isFraud", "datetime"))

train_matrix <- data.matrix(train_set[, ..features])
val_matrix   <- data.matrix(val_set[, ..features])

dtrain <- xgb.DMatrix(data = train_matrix, label = train_labels)
dval   <- xgb.DMatrix(data = val_matrix, label = val_labels)

params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "aucpr",
  eta = 0.05,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8,
  scale_pos_weight = sum(train_labels == 0) / sum(train_labels == 1)
)

k <- 5
set.seed(123)
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

final_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,
  watchlist = list(train = dtrain, val = dval),
  early_stopping_rounds = 50,
  maximize = TRUE,
  verbose = 1
)

xgb.save(final_model, "models/final_xgb_model_cv.model")
cat("Model saved as final_xgb_model_cv.model\n")