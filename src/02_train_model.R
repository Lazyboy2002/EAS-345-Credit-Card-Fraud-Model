library(xgboost)

# Load preprocessed data. Only load train_set and val_set.
#do not train model on test set
train_set <- readRDS("data/train_set.rds")
val_set   <- readRDS("data/val_set.rds")

# Ensure everything is a flat data frame
train_set <- as.data.frame(train_set)
val_set   <- as.data.frame(val_set)

# Get a table that contains all columns except the target column isFraud
# these are all the predictor variables
X_train <- train_set[, !names(train_set) %in% "isFraud"]

#extract the isFraud column from train set
y_train <- train_set$isFraud

#as above for validation set
X_val <- val_set[, !names(val_set) %in% "isFraud"]
y_val <- val_set$isFraud

# Convert all columns to numeric (handle POSIXct or logical if any) in train set
X_train[] <- lapply(X_train, function(col) {
  if (inherits(col, "POSIXct") || inherits(col, "POSIXt")) as.numeric(col)
  else as.numeric(col)
})

# Convert all columns to numeric (handle POSIXct or logical if any) in val set
X_val[] <- lapply(X_val, function(col) {
  if (inherits(col, "POSIXct") || inherits(col, "POSIXt")) as.numeric(col)
  else as.numeric(col)
})

# Convert each table to an xgb.DMatrix
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dval   <- xgb.DMatrix(data = as.matrix(X_val), label = y_val)

# Set XGBoost parameters

#objective - binary classification problem
#eval_metric - evaluation metric is area under ROC curve
#max depth - max depth of a decision tree
#eta - learning rate
#subsample - fraction of training data randomly sampled for growing each tree
#colsample_bytree - fraction of features randomly sampled for each tree
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train the model with xgb.train()
# params - the model parameters needed
# data - the training data, an xgb.DMatrix
# nrounds - max number of iterations (trees) to build the model
# watchlist - which datasets to monitor during training. observe model perf in real time
# early_stopping_round - if validation AUC has not improved for 30 rounds, stop training
# print_every_n - training progress printed to console every n rounds
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 300,
  watchlist = list(train = dtrain, val = dval),
  early_stopping_rounds = 30,
  print_every_n = 10
)

# Save the trained model to the created directory

#dir.create("output", showWarnings = FALSE)
#xgb.save(xgb_model, "output/xgb_fraud.model")

#print feature importance matrix, which features were most useful for making predictions
importance_matrix <- xgb.importance(model = xgb_model)
print(importance_matrix[1:10, ])  # top 10 features

#plot feature importance
xgb.plot.importance(importance_matrix)
