library(data.table)   # for fast I/O and joining
library(dplyr)        # for data manipulation
library(ggplot2)      # optional, for quick exploration
library(caret)
library(fastDummies)
library(lubridate)
library(xgboost)
library(smotefamily)

setwd("C:\\Users\\loghe\\OneDrive\\Desktop\\school\\buffalo\\2025-2026\\fall-2025\\courses\\eas345-intro-data-science\\project\\group_project_repo\\dataset\\ieee-fraud-detection")

#STEP 1: DATA INTEGRATION
#STEP 1: DATA INTEGRATION
#STEP 1: DATA INTEGRATION

# Load using fread for speed
train_transaction <- fread("train_transaction.csv")
train_identity    <- fread("train_identity.csv")

#str(train_transaction)
#str(train_identity)

# Check primary key overlap
#intersect(names(train_transaction), names(train_identity))

# Left join for train set
train <- merge(train_transaction, train_identity,
               by = "TransactionID",
               all.x = TRUE)

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


if (FALSE) {
  #create a csv of train, val, and test
  #RUN THIS CODE ONE TIME
  #RUN THIS CODE ONE TIME
  #fwrite(train_set, "train_set_processed.csv")
  #fwrite(val_set, "validation_set_processed.csv")
  #fwrite(test_set, "test_set_processed.csv")
  
  #compute missingn percentage from the training set only
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
  
  # Check that all have the same number of columns
  #ncol(train_set)
  #ncol(validation_clean)
  #ncol(test_set)
  
  #Check that all column names match
  #identical(names(train_set), names(validation_clean))
  #identical(names(train_set), names(test_set))
  
  
  #STEP 2: OUTLIER AND ZERO TRANSACTION HANDLING
  #STEP 2: OUTLIER AND ZERO TRANSACTION HANDLING
  #STEP 2: OUTLIER AND ZERO TRANSACTION HANDLING
  
  #record the number of rows before we remove invalid transactionAmt
  before_rows_train <- nrow(train_set)
  before_rows_val <- nrow(val_set)
  before_rows_test <- nrow(test_set)
  
  #remove transactions with transactionAmt == 0. likely
  #invalid transactions
  train_set <- train_set[TransactionAmt != 0]
  val_set <- val_set[TransactionAmt != 0]
  test_set <- test_set[TransactionAmt != 0]
  
  #record the number of rows after we remove invalid transactionAmt
  after_rows_train <- nrow(train_set)
  after_rows_val <- nrow(val_set)
  after_rows_test <- nrow(test_set)
  
  #amount of rows removed
  removed_rows_train <- before_rows_train - after_rows_train
  removed_rows_val <- before_rows_val - after_rows_val
  removed_rows_test <- before_rows_test - after_rows_test
  
  
  #use box plot to visualize transactionAmt, to view any outliers
  #ggplot(train_set, aes(x = "", y = TransactionAmt)) +
  #  geom_boxplot(outlier.color = "red", fill = "lightblue") +
  #  scale_y_continuous(trans = 'log10') +
  #  labs(title = "Boxplot of Transaction Amounts (Log Scale)",
  #       y = "Transaction Amount (log10)",
  #       x = "")
  
  library(ggplot2)
  
  
  #use log scaled histogram to visuzalize transactionAmt, to view any outliers, and skewedness
  #x intercept at the 99th percentile mark to show outliers
  q99 <- quantile(train_set$TransactionAmt, 0.99, na.rm = TRUE)
  
  #ggplot(train_set, aes(x = TransactionAmt)) +
  #  geom_histogram(bins = 100, fill = "steelblue", color = "black", alpha = 0.7) +
  #  geom_vline(xintercept = q99, color = "red", linetype = "dashed", linewidth = 1) +
  #  scale_x_continuous(trans = "log10") +
  #  labs(
  #    title = "Transaction Amounts with 99th Percentile Marked",
  #    x = "Transaction Amount (log10 scale)",
  #    y = "Frequency"
  #  ) +
  #  theme_minimal()
  
  
  #flag the top 1% of transactions in new column is_outlier_amt
  upper_threshold <- quantile(train_set$TransactionAmt, 0.99, na.rm = TRUE)
  lower_threshold <- quantile(train_set$TransactionAmt, 0.01, na.rm = TRUE)
  
  train_set[, is_outlier_amt := ifelse(TransactionAmt > upper_threshold | TransactionAmt < lower_threshold, 1, 0)]
  val_set[, is_outlier_amt := ifelse(TransactionAmt > upper_threshold | TransactionAmt < lower_threshold, 1, 0)]
  test_set[, is_outlier_amt := ifelse(TransactionAmt > upper_threshold | TransactionAmt < lower_threshold, 1, 0)]
  
  #count and display the number of transactions flagged as outliers
  cat("Flagged", sum(train_set$is_outlier_amt), "transactions as outliers (",
      round(mean(train_set$is_outlier_amt) * 100, 3), "% of data)\n")
  
  
  
  #STEP 3: MISSING VALUE TREATMENT
  #STEP 3: MISSING VALUE TREATMENT
  #STEP 3: MISSING VALUE TREATMENT
  
  
  #view all the numeric columns in the dataset
  numeric_cols <- names(train_set)[sapply(train_set, is.numeric)]
  
  #only view from the rows that are not fraud, as they will have different
  #distribution than rows that are fraud
  numeric_cols <- setdiff(numeric_cols, "isFraud")  # Exclude target variable
  
  
  #create columns for each numerical column to specify if
  #row had missing value in that column or not. roughly doubles
  #the size of the table. columns are 0 or 1, 0 if row had
  #value in column, 1 if row had missing value in column
  for (col in numeric_cols) {
    flag_col <- paste0(col, "_missing")
    
    #add missing flags for each dataset
    train_set[, (flag_col) := ifelse(is.na(get(col)), 1, 0)]
    val_set[, (flag_col) := ifelse(is.na(get(col)), 1, 0)]
    test_set[, (flag_col) := ifelse(is.na(get(col)), 1, 0)]
  }
  
  numeric_cols <- setdiff(names(train_set)[sapply(train_set, is.numeric)], "isFraud")
  
  # Compute medians per class for all numeric columns at once
  medians_per_class <- train_set[, lapply(.SD, function(x) median(x, na.rm = TRUE)), by = isFraud, .SDcols = numeric_cols]
  
  
  print(medians_per_class)
  
  #loop the numeric columns and set the missing values to the median of each column
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
  
  
  
  ncol(train_set)
  ncol(val_set)
  ncol(test_set)
  #head(train_set)
  #View(train_set)
  
  
  # Count remaining missing values in numeric columns for each dataset
  na_summary <- data.table(
    Dataset = c("Train", "Validation", "Test"),
    Missing_Count = c(
      sum(sapply(train_set[, ..numeric_cols], function(x) any(is.na(x)))),
      sum(sapply(val_set[, ..numeric_cols], function(x) any(is.na(x)))),
      sum(sapply(test_set[, ..numeric_cols], function(x) any(is.na(x))))
    )
  )
  
  #print(na_summary)
  
  
  
  #IMPUTE MISSING CATEGORICAL VARS WITH MISSING
  #IMPUTE MISSING CATEGORICAL VARS WITH MISSING
  #IMPUTE MISSING CATEGORICAL VARS WITH MISSING
  
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
  
  
  cat_cols_to_impute
  
  #verify no NAs remain
  #sapply(train_set[, ..cat_cols_to_impute], function(x) sum(is.na(x)))
  #sapply(validation_set[, ..cat_cols_to_impute], function(x) sum(is.na(x)))
  #sapply(test_set[, ..cat_cols_to_impute], function(x) sum(is.na(x)))
  
  
  
  #LABEL ENCODE HIGH CARDINALITY CATEGORICAL FEATURES
  #LABEL ENCODE HIGH CARDINALITY CATEGORICAL FEATURES
  #LABEL ENCODE HIGH CARDINALITY CATEGORICAL FEATURES
  
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
  
  
  
  #ONE HOT ENCODE LOW CARDINALITY CATEGORICAL
  #ONE HOT ENCODE LOW CARDINALITY CATEGORICAL
  #ONE HOT ENCODE LOW CARDINALITY CATEGORICAL
  
  low_card_cols <- cat_cols[sapply(train_set[, ..cat_cols], function(x) length(unique(x)) < 10)]
  
  #View the low cardinality columns
  low_card_cols
  
  train_set <- dummy_cols(
    train_set,
    select_columns = low_card_cols,
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
  
  
  # 1. Get all unique column names across datasets
  all_cols <- union(names(train_set), union(names(val_set), names(test_set)))
  
  setDT(train_set)
  setDT(val_set)
  setDT(test_set)
  
  # Add missing columns with zeros safely
  for (col in setdiff(all_cols, names(train_set))) {
    set(train_set, j = col, value = 0)
  }
  for (col in setdiff(all_cols, names(val_set))) {
    set(val_set, j = col, value = 0)
  }
  for (col in setdiff(all_cols, names(test_set))) {
    set(test_set, j = col, value = 0)
  }
  
  # 3. Reorder columns to be consistent
  setcolorder(val_set, names(train_set))
  setcolorder(test_set, names(train_set))
  
  #grep("_", names(train_set), value = TRUE)   # list dummy columns
  #ncol(train_set)
  
  #setequal(names(train_set), names(val_set))   # TRUE
  #setequal(names(train_set), names(test_set))
  
  
  
  
  
  
  #FEATURE ENGINEERING
  #FEATURE ENGINEERING
  #FEATURE ENGINEERING
  
  
  
  #TEMPORAL FEATURES AND USER LEVEL AGGREGATES
  #TEMPORAL FEATURES AND USER LEVEL AGGREGATES
  #TEMPORAL FEATURES AND USER LEVEL AGGREGATES
  # Define the starting point
  origin_date <- as.POSIXct("2017-12-01", tz = "UTC")
  
  # List of datasets
  datasets <- list(train_set, val_set, test_set)
  
  # Loop over datasets
  for (dt in datasets) {
    
    # 1. Create datetime and days
    dt[, datetime := origin_date + TransactionDT]
    dt[, days := TransactionDT / (24*60*60)]
    
    # 2. Sort by card1 and datetime (needed for rolling calculations)
    setorder(dt, card1, datetime)
    
    # 3. Temporal features
    dt[, hour := hour(datetime)]
    dt[, day_of_week := wday(datetime, label = FALSE) - 1]  # 0 = Monday
    dt[, day_of_month := mday(datetime)]
    dt[, week_of_year := isoweek(datetime)]
    
    # 4. Transaction velocity features
    dt[, transactions_per_day := .N, by = .(card1, day_of_month, week_of_year)]
    dt[, transactions_per_hour := .N, by = .(card1, day_of_week, hour)]
    
    # 5. User-level aggregates per card1
    dt[, mean_amt := mean(TransactionAmt, na.rm = TRUE), by = card1]
    dt[, std_amt  := sd(TransactionAmt,  na.rm = TRUE), by = card1]
    
    # Rolling transaction count over 7-day window
    dt[, tx_count_7d := sapply(seq_len(.N), function(i) {
      sum(days >= days[i] - 7 & days <= days[i])
    }), by = card1]
    
    # Rolling std of amount over last 30 days
    dt[, amt_std_30d := sapply(seq_len(.N), function(i) {
      idx <- which(days >= days[i] - 30 & days <= days[i])
      sd(TransactionAmt[idx], na.rm = TRUE)
    }), by = card1]
    
    # Unique email domains in last 30 days (if present)
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
  
  
  
  
  
  #LOG TRANSFORMAION
  #LOG TRANSFORMAION
  #LOG TRANSFORMAION
  for (data in list(train_set, val_set, test_set)) {
    data[, TransactionAmt_log := log1p(TransactionAmt)]  # log1p(x) = log(1 + x)
  }
  
  summary(train_set$TransactionAmt_log)
  
  # Optional: compare original vs transformed distributions
  hist(train_set$TransactionAmt, breaks = 50, main = "Original TransactionAmt", xlab = "Amount")
  hist(train_set$TransactionAmt_log, breaks = 50, main = "Log-Transformed TransactionAmt", xlab = "log(1 + Amount)")
  
  
  
  
  
  #FEATURE SELECTION
  #FEATURE SELECTION
  #FEATURE SELECTION
  
  
  #Drop one variable in any pair with Pearson correlation > 0.9.
  #Drop one variable in any pair with Pearson correlation > 0.9.
  #Drop one variable in any pair with Pearson correlation > 0.9.
  
  # Exclude identifiers from numeric columns
  id_cols <- c("TransactionID", "TransactionDT")
  num_cols <- setdiff(names(train_set)[sapply(train_set, is.numeric)], id_cols)
  
  # Create train_num_clean without IDs
  train_num_clean <- train_set[, ..num_cols]
  
  # Sample rows to compute correlation matrix
  set.seed(42)
  sample_idx <- sample(nrow(train_num_clean), min(50000, nrow(train_num_clean)))
  train_sample <- train_num_clean[sample_idx, ]
  
  # Remove columns with zero variance in the sample
  non_constant_cols <- names(train_sample)[sapply(train_sample, function(x) var(x, na.rm = TRUE) > 0)]
  train_sample_clean <- train_sample[, ..non_constant_cols]
  
  corr_matrix_clean <- cor(train_sample_clean, use = "pairwise.complete.obs")
  
  #View(corr_matrix_clean)
  
  #columns that are highly correlated
  high_corr <- findCorrelation(corr_matrix_clean, cutoff = 0.9, names = TRUE)
  
  high_corr
  
  cols_to_drop <- intersect(high_corr, names(train_set))
  
  #drop the columns from each dataset
  train_set[, (cols_to_drop) := NULL]
  val_set[, (cols_to_drop) := NULL]
  test_set[, (cols_to_drop) := NULL]
  
  
  #test that features are the same across all datasets
  setequal(names(train_set), names(val_set))  # should be TRUE
  setequal(names(train_set), names(test_set)) # should be TRUE
}




View(train_set)
View(val_set)
View(test_set)