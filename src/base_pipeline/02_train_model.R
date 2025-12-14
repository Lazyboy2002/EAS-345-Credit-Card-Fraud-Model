library(xgboost)

# Load preprocessed data. Only load train_set and val_set.
#do not train model on test set
train_set <- readRDS("data/processed/train_set.rds")
val_set   <- readRDS("data/processed/val_set.rds")

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

#set the number of folds for cross validation
k <- 5

#set seed to ensure reproducibility
set.seed(123)

#create a list of row indicies for each fold
#seq_len(nrow(train_matrix)) - sequence of row indicies from 1 to number of training samples
#rep(1:k, length.out = nrow(train_matrix)) - repeats fold numbers 1-k until all rows are assigned
#sample(...) - randomly shuffle fold assignments
#split(..., ...) - separates row indices into k groups, stored in a list
#where each element corresponds to one fold
cv_folds <- split(seq_len(nrow(dtrain)), sample(rep(1:k, length.out = nrow(dtrain))))

#call to the xgboost cross validation function
#params = params, - use the previously defined hyperparameters
#data = dtrain - specify the training data in DMatrix format
#nrounds = 2000 - max number of trees to train
#folds = cv_folds - manual list of fold indices for cross validation
#early_stopping_rounds = 50 - stop training if the evaluation metric does not improve for 50 consecutive rounds
#maximize = TRUE - tell XGBoost to maximize the evaluation metric
#verbose = 1 - tell xgboost to print training progress to console
cv_model <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 2000,
  folds = cv_folds,
  early_stopping_rounds = 50,
  maximize = TRUE,
  verbose = 1
)

#retrieve the optimal number of boosting rounds from the cross validation results
#used to train the final model with the best number of trees
best_nrounds <- cv_model$best_iteration

# Train the model with xgb.train()
# params - the model parameters needed
# data - the training data, an xgb.DMatrix
#nrounds = best_nrounds - train for the optimal number of boosting rounds from CV
# watchlist - which datasets to monitor during training. observe model perf in real time
# early_stopping_round - if validation AUC has not improved for 30 rounds, stop training
# print_every_n - training progress printed to console every n rounds
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,
  watchlist = list(train = dtrain, val = dval),
  early_stopping_rounds = 30,
  print_every_n = 10
)

# Save the trained model to the created directory
xgb.save(xgb_model, "models/xgb_fraud.model")

#print feature importance matrix, which features were most useful for making predictions
importance_matrix <- xgb.importance(model = xgb_model)
print(importance_matrix[1:10, ])  # top 10 features

#plot feature importance
xgb.plot.importance(importance_matrix)
