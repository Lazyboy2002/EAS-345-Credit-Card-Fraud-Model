EAS 345 Credit Card Fraud Model


Project Overview

	Title: EAS 345 Credit Card Fraud Model

	Goal: Detect credit card fraud using machine learning models.

Team Information

	Team Name: Group 1

	Primary Repository Maintainer: Matthew Drummond

	Team Members and Roles:

		Dan Caverly – Project Lead
		Matthew Drummond – Data Architect
		Michael Hicks – Data Scientist
		Sean Stack – Communication Liaison

Data

	Dataset: IEEE-CIS Fraud Detection dataset

	Source: Provisioned by IEEE-CIS for a Kaggle competition

	Storage

		RDS versions of the raw datasets (train_transaction.rds and train_identity.rds) are stored in data/raw/ inside this repository.

		Everything needed to run the project is included — no external downloads are required.

Repository Structure

	data/
		raw/
			train_identity.rds
			train_transaction.rds
		processed/
			test_set.rds
			train_set.rds
			val_set.rds
	models/
		xgb_fraud.model
		final_xgb_model_cv.model
	src/
		01_preprocess.R
		02_train_model.R
		03_test_model.R
	README.md

How to Train and Test the Base Model

	1. In R Studio, set your working directory to the project root

	2. Preprocess the data:

		Run src/base_pipeline/01_preprocess.R in R Studio

	3. Train the model

		Run src/base_pipeline/02_train_model.R in R Studio

	4. Test/evaluate the model

		Run src/base_pipeline/03_test_model.R in R Studio

Required R Packages

	The following R packages are required to run the project:

		data.table
		dplyr
		ggplot2
		caret
		fastDummies
		lubridate
		xgboost
		smotefamily
		pROC
		PRROC

	Installation Note:

		Install missing packages using the RStudio package installer, or run in R:

			install.packages(c("data.table","dplyr","ggplot2","caret","fastDummies","lubridate","xgboost","smotefamily","pROC","PRROC"))

Notes

	The project is fully reproducible; all data and scripts are included.

	The RDS format ensures fast loading and exact preservation of data types.

	Always run scripts in the order listed above to avoid errors.