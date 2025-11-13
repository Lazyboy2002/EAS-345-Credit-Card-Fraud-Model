library(data.table)
library(dplyr)
library(ggplot2)
library(caret)
library(fastDummies)
library(lubridate)
library(xgboost)
library(smotefamily)





#STEP 1: DATA INTEGRATION
#STEP 1: DATA INTEGRATION
#STEP 1: DATA INTEGRATION

# Load data using fread for speed
train_transaction <- fread("train_transaction.csv")
train_identity    <- fread("train_identity.csv")



# STEP 1.1: Left join transaction and identity tables
# STEP 1.1: Left join transaction and identity tables
# STEP 1.1: Left join transaction and identity tables
train <- merge(train_transaction, train_identity,
               by = "TransactionID",
               all.x = TRUE)

#seed for random number replication
set.seed(123)

# Create index for 70% train data. createDataPartition returns the index for splitting the training set
trainIndex <- createDataPartition(train$isFraud, p = 0.7, list = FALSE)

#perform the split into train set and store validation and test set in temp set
train_set <- train[trainIndex, ]
temp_set  <- train[-trainIndex, ]

# Split the remaining 30% equally for validation and test. using temp set here
valIndex <- createDataPartition(temp_set$isFraud, p = 0.5, list = FALSE)

#perform the split of temp set to get validation set and test set
val_set  <- temp_set[valIndex, ]
test_set <- temp_set[-valIndex, ]




#STEP 1.2: DROP COLUMNS WITH > 95% MISSING VALUES
#STEP 1.2: DROP COLUMNS WITH > 95% MISSING VALUES
#STEP 1.2: DROP COLUMNS WITH > 95% MISSING VALUES

#compute missing percentage from the training set only
missing_pct_train <- colMeans(is.na(train_set)) * 100
#missing_pct_train

#identify columns to keep from the dataset
cols_to_keep <- names(missing_pct_train[missing_pct_train <= 95])
#cols_to_keep

#subset all datasets using the same columns. clean datasets have removed all columns
#with > 95% missing values
train_set <- train_set[, ..cols_to_keep]
val_set <- val_set[, ..cols_to_keep]
test_set <- test_set[, ..cols_to_keep]





#STEP 2: OUTLIER AND ZERO TRANSACTION HANDLING
#STEP 2: OUTLIER AND ZERO TRANSACTION HANDLING
#STEP 2: OUTLIER AND ZERO TRANSACTION HANDLING


#STEP 2.1: REMOVE TRANSACTIONS WITH TRANSACTIONAMT == 0
#STEP 2.1: REMOVE TRANSACTIONS WITH TRANSACTIONAMT == 0
#STEP 2.1: REMOVE TRANSACTIONS WITH TRANSACTIONAMT == 0

#remove transactions with transactionAmt == 0. likely invalid transactions
train_set <- train_set[TransactionAmt != 0]
val_set <- val_set[TransactionAmt != 0]
test_set <- test_set[TransactionAmt != 0]



#STEP 2.2: FLAG TOP 1% TRANSACTION AMOUNTS
#STEP 2.2: FLAG TOP 1% TRANSACTION AMOUNTS
#STEP 2.2: FLAG TOP 1% TRANSACTION AMOUNTS

#flag the top 1% of transactions in new column is_outlier_amt
upper_threshold <- quantile(train_set$TransactionAmt, 0.99, na.rm = TRUE)
lower_threshold <- quantile(train_set$TransactionAmt, 0.01, na.rm = TRUE)

#add the new column to each dataframe
train_set[, is_outlier_amt := ifelse(TransactionAmt > upper_threshold | TransactionAmt < lower_threshold, 1, 0)]
val_set[, is_outlier_amt := ifelse(TransactionAmt > upper_threshold | TransactionAmt < lower_threshold, 1, 0)]
test_set[, is_outlier_amt := ifelse(TransactionAmt > upper_threshold | TransactionAmt < lower_threshold, 1, 0)]





#STEP 3: MISSING VALUE TREATMENT
#STEP 3: MISSING VALUE TREATMENT
#STEP 3: MISSING VALUE TREATMENT





#view all the numeric columns in the dataset
numeric_cols <- names(train_set)[sapply(train_set, is.numeric)]

#only view from the rows that are not fraud, as they will have different
#distribution than rows that are fraud
numeric_cols <- setdiff(numeric_cols, "isFraud")  # Exclude target variable


#STEP 3.1: CREATE BINARY FEATURE MISSING INDICATOR
#STEP 3.1: CREATE BINARY FEATURE MISSING INDICATOR
#STEP 3.1: CREATE BINARY FEATURE MISSING INDICATOR

#create columns for each numerical column to specify if
#row had missing value in that column or not. roughly doubles
#the size of the table. columns are 0 or 1, 0 if row had
#value in column, 1 if row had missing value in column
for (col in numeric_cols) {
  flag_col <- paste0(col, "_missing")
  
  #add missing flags to each dataset
  train_set[, (flag_col) := ifelse(is.na(get(col)), 1, 0)]
  val_set[, (flag_col) := ifelse(is.na(get(col)), 1, 0)]
  test_set[, (flag_col) := ifelse(is.na(get(col)), 1, 0)]
}

numeric_cols <- setdiff(names(train_set)[sapply(train_set, is.numeric)], "isFraud")



#STEP 3.2: IMPUTE PER CLASS USING MEDIAN
#STEP 3.2: IMPUTE PER CLASS USING MEDIAN
#STEP 3.2: IMPUTE PER CLASS USING MEDIAN

# Compute medians per class for all numeric columns at once
medians_per_class <- train_set[, lapply(.SD, function(x) median(x, na.rm = TRUE)), by = isFraud, .SDcols = numeric_cols]

#loop the numeric columns and set the missing values to the median of each column
#median by class
for (col in numeric_cols) {
  med_0 <- medians_per_class[isFraud == 0, get(col)]
  med_1 <- medians_per_class[isFraud == 1, get(col)]
  
  #TRAIN SET
  idx0 <- which(is.na(train_set[[col]]) & train_set$isFraud == 0)
  idx1 <- which(is.na(train_set[[col]]) & train_set$isFraud == 1)
  if (length(idx0) > 0) set(train_set, i = idx0, j = col, value = med_0)
  if (length(idx1) > 0) set(train_set, i = idx1, j = col, value = med_1)
  
  #VALIDATION SET
  idx0 <- which(is.na(val_set[[col]]) & val_set$isFraud == 0)
  idx1 <- which(is.na(val_set[[col]]) & val_set$isFraud == 1)
  if (length(idx0) > 0) set(val_set, i = idx0, j = col, value = med_0)
  if (length(idx1) > 0) set(val_set, i = idx1, j = col, value = med_1)
  
  #TEST SET
  idx0 <- which(is.na(test_set[[col]]) & test_set$isFraud == 0)
  idx1 <- which(is.na(test_set[[col]]) & test_set$isFraud == 1)
  if (length(idx0) > 0) set(test_set, i = idx0, j = col, value = med_0)
  if (length(idx1) > 0) set(test_set, i = idx1, j = col, value = med_1)
}



# Count remaining missing values in numeric columns for each dataset
na_summary <- data.table(
  Dataset = c("Train", "Validation", "Test"),
  Missing_Count = c(
    sum(sapply(train_set[, ..numeric_cols], function(x) any(is.na(x)))),
    sum(sapply(val_set[, ..numeric_cols], function(x) any(is.na(x)))),
    sum(sapply(test_set[, ..numeric_cols], function(x) any(is.na(x))))
  )
)



#STEP 3.3: IMPUTE MISSING CATEGORICAL VARS WITH MISSING
#STEP 3.3: IMPUTE MISSING CATEGORICAL VARS WITH MISSING
#STEP 3.3: IMPUTE MISSING CATEGORICAL VARS WITH MISSING

#store the categorical columns in a list
cat_cols <- names(train_set)[sapply(train_set, is.character) | sapply(train_set, is.factor)]

#Calculate missing percentage per categorical feature
missing_pct_cat <- sapply(train_set[, ..cat_cols], function(x) mean(is.na(x)))

#Keep columns with <95% missing
cat_cols_to_impute <- names(missing_pct_cat[missing_pct_cat < 0.95])

#Replace NAs with "missing" in train, validation, and test sets
for (col in cat_cols_to_impute) {
  for (dataset in list(train_set, val_set, test_set)) {
    # If factor, ensure "missing" is a valid level
    if (is.factor(dataset[[col]])) {
      levels(dataset[[col]]) <- union(levels(dataset[[col]]), "missing")
    }
    
    # Replace NA or empty strings with "missing"
    idx <- which(is.na(dataset[[col]]) | dataset[[col]] == "")
    set(dataset, i = idx, j = col, value = "missing")
  }
}




#STEP 4: ENCODING
#STEP 4: ENCODING
#STEP 4: ENCODING

#STEP 4.1: LABEL ENCODE HIGH CARDINALITY CATEGORICAL FEATURES
#STEP 4.1: LABEL ENCODE HIGH CARDINALITY CATEGORICAL FEATURES
#STEP 4.1: LABEL ENCODE HIGH CARDINALITY CATEGORICAL FEATURES

#check which features are high cardinality features
cardinality <- sapply(train_set[, ..cat_cols], function(x) length(unique(x)))
cardinality[order(-cardinality)]  # sorted from highest to lowest

#list of high cardinality categorical columns
high_card_cols <- c("DeviceInfo", "id_33", "id_31", "id_30", "R_emaildomain", "P_emaildomain")

#loop high cardinality categorical features and do label encoding
for (col in high_card_cols) {
  
  # Create mapping from unique category to integer
  all_levels <- unique(c(train_set[[col]], val_set[[col]], test_set[[col]]))
  mapping <- setNames(seq_along(all_levels), all_levels)
  
  # Apply mapping to all datasets
  for (dataset in list(train_set, val_set, test_set)) {
    dataset[, (col) := mapping[dataset[[col]]]]
  }
}



#STEP 4.2: ONE HOT ENCODE LOW CARDINALITY CATEGORICAL
#STEP 4.2: ONE HOT ENCODE LOW CARDINALITY CATEGORICAL
#STEP 4.2: ONE HOT ENCODE LOW CARDINALITY CATEGORICAL

#create a list of low cadinality categorical features
low_card_cols <- cat_cols[sapply(train_set[, ..cat_cols], function(x) length(unique(x)) < 10)]


#Create new dummy columns for each category level.
train_set <- dummy_cols(
  
  #on train_set dataset
  train_set,
  
  #specifies which columns to encode (low_card_cols).
  select_columns = low_card_cols,
  
  #drop the original categorical column
  remove_selected_columns = TRUE
)

val_set <- dummy_cols(
  val_set,
  select_columns = low_card_cols,
  remove_selected_columns = TRUE
)

test_set <- dummy_cols(
  test_set,
  select_columns = low_card_cols,
  remove_selected_columns = TRUE
)


#to make sure train_set, val_set, and test_set have the same columns
#some one hot encode features only appear in one table

# 1. Get all unique column names across datasets
all_cols <- union(names(train_set), union(names(val_set), names(test_set)))

#make sure the tables are data.tables
setDT(train_set)
setDT(val_set)
setDT(test_set)

# Add missing columns with zeros safely, columns that were only in one table, but not in the others
for (col in setdiff(all_cols, names(train_set))) {
  set(train_set, j = col, value = 0)
}
for (col in setdiff(all_cols, names(val_set))) {
  set(val_set, j = col, value = 0)
}
for (col in setdiff(all_cols, names(test_set))) {
  set(test_set, j = col, value = 0)
}

#Reorder columns to be consistent
setcolorder(val_set, names(train_set))
setcolorder(test_set, names(train_set))







#STEP 5: FEATURE ENGINEERING
#STEP 5: FEATURE ENGINEERING
#STEP 5: FEATURE ENGINEERING

#STEP 5.1: TEMPORAL FEATURES AND USER LEVEL AGGREGATES
#STEP 5.1: TEMPORAL FEATURES AND USER LEVEL AGGREGATES
#STEP 5.1: TEMPORAL FEATURES AND USER LEVEL AGGREGATES

# Define the starting point
origin_date <- as.POSIXct("2017-12-01", tz = "UTC")

# List of datasets
datasets <- list(train_set, val_set, test_set)

# Loop over datasets
for (dt in datasets) {
  
  #Create datetime and days
  dt[, datetime := origin_date + TransactionDT]
  dt[, days := TransactionDT / (24*60*60)]
  
  #Sort by card1 and datetime (needed for rolling calculations)
  setorder(dt, card1, datetime)
  
  #create Temporal features in each dataset
  dt[, hour := hour(datetime)]
  dt[, day_of_week := wday(datetime, label = FALSE) - 1]  # 0 = Monday
  dt[, day_of_month := mday(datetime)]
  dt[, week_of_year := isoweek(datetime)]
  
  #create Transaction velocity features in each dataset
  dt[, transactions_per_day := .N, by = .(card1, day_of_month, week_of_year)]
  dt[, transactions_per_hour := .N, by = .(card1, day_of_week, hour)]
  
  #create User-level aggregates per card1 in each dataset
  dt[, mean_amt := mean(TransactionAmt, na.rm = TRUE), by = card1]
  dt[, std_amt  := sd(TransactionAmt,  na.rm = TRUE), by = card1]
  
  #create Rolling transaction count over 7-day window feature
  dt[, tx_count_7d := sapply(seq_len(.N), function(i) {
    sum(days >= days[i] - 7 & days <= days[i])
  }), by = card1]
  
  #create Rolling std of amount over last 30 days feature
  dt[, amt_std_30d := sapply(seq_len(.N), function(i) {
    idx <- which(days >= days[i] - 30 & days <= days[i])
    sd(TransactionAmt[idx], na.rm = TRUE)
  }), by = card1]
  
  #create Unique email domains in last 30 days feature
  if (all(c("P_emaildomain", "R_emaildomain") %in% names(dt))) {
    dt[, unique_emaildomain_30d := sapply(seq_len(.N), function(i) {
      idx <- which(days >= days[i] - 30 & days <= days[i])
      length(unique(c(P_emaildomain[idx], R_emaildomain[idx])))
    }), by = card1]
  }
}

#handle the NAs that were introduced in standard deviation columns
#replace NA with 0
for (dt in list(train_set, val_set, test_set)) {
  dt[is.na(std_amt), std_amt := 0]
  dt[is.na(amt_std_30d), amt_std_30d := 0]
}





#STEP 5.2: LOG TRANSFORMAION
#STEP 5.2: LOG TRANSFORMAION
#STEP 5.2: LOG TRANSFORMAION
for (data in list(train_set, val_set, test_set)) {
  data[, TransactionAmt_log := log1p(TransactionAmt)]  # log1p(x) = log(1 + x)
}





#STEP 6: FEATURE SELECTION
#STEP 6: FEATURE SELECTION
#STEP 6: FEATURE SELECTION


#STEP 6.1: Drop one variable in any pair with Pearson correlation > 0.9.
#STEP 6.1: Drop one variable in any pair with Pearson correlation > 0.9.
#STEP 6.1: Drop one variable in any pair with Pearson correlation > 0.9.

# Exclude identifiers from numeric columns
id_cols <- c("TransactionID", "TransactionDT")
num_cols <- setdiff(names(train_set)[sapply(train_set, is.numeric)], id_cols)

# Create train_num_clean without IDs
train_num_clean <- train_set[, ..num_cols]

# Sample rows to compute correlation matrix. Set seed for reproducibility
set.seed(42)

#sample only 50000 rows so code can complete in timely fashion
sample_idx <- sample(nrow(train_num_clean), min(50000, nrow(train_num_clean)))

#create the random sample
train_sample <- train_num_clean[sample_idx, ]

# Remove columns with zero variance in the sample
non_constant_cols <- names(train_sample)[sapply(train_sample, function(x) var(x, na.rm = TRUE) > 0)]
train_sample_clean <- train_sample[, ..non_constant_cols]

#compute the correlation matrix
corr_matrix_clean <- cor(train_sample_clean, use = "pairwise.complete.obs")

#get list of the columns that are highly correlated
high_corr <- findCorrelation(corr_matrix_clean, cutoff = 0.9, names = TRUE)

#section out the features to drop
cols_to_drop <- intersect(high_corr, names(train_set))

#drop the columns from each dataset
train_set[, (cols_to_drop) := NULL]
val_set[, (cols_to_drop) := NULL]
test_set[, (cols_to_drop) := NULL]



View(train_set)
View(val_set)
View(test_set)

