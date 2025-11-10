library(xgboost)

# Load preprocessed data
train_set <- readRDS("C:\\Users\\loghe\\OneDrive\\Desktop\\school\\buffalo\\2025-2026\\fall-2025\\courses\\eas345-intro-data-science\\project\\group_project_repo\\dataset\\ieee-fraud-detection\\train_set.rds")
val_set   <- readRDS("C:\\Users\\loghe\\OneDrive\\Desktop\\school\\buffalo\\2025-2026\\fall-2025\\courses\\eas345-intro-data-science\\project\\group_project_repo\\dataset\\ieee-fraud-detection\\val_set.rds")

# Ensure everything is a flat data frame
train_set <- as.data.frame(train_set)
val_set   <- as.data.frame(val_set)

# Separate features and target
X_train <- train_set[, !names(train_set) %in% "isFraud"]
y_train <- train_set$isFraud

X_val <- val_set[, !names(val_set) %in% "isFraud"]
y_val <- val_set$isFraud

# Convert all columns to numeric (handle POSIXct or logical if any)
X_train[] <- lapply(X_train, function(col) {
  if (inherits(col, "POSIXct") || inherits(col, "POSIXt")) as.numeric(col)
  else as.numeric(col)
})

X_val[] <- lapply(X_val, function(col) {
  if (inherits(col, "POSIXct") || inherits(col, "POSIXt")) as.numeric(col)
  else as.numeric(col)
})

# Convert to xgb.DMatrix
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dval   <- xgb.DMatrix(data = as.matrix(X_val), label = y_val)

# Set XGBoost parameters
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train the model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 300,
  watchlist = list(train = dtrain, val = dval),
  early_stopping_rounds = 30,
  print_every_n = 10
)

# Save the trained model
dir.create("output", showWarnings = FALSE)
xgb.save(xgb_model, "output/xgb_fraud.model")

# Optional: print feature importance
importance_matrix <- xgb.importance(model = xgb_model)
print(importance_matrix[1:10, ])  # top 10 features

# Optional: plot feature importance
xgb.plot.importance(importance_matrix)
