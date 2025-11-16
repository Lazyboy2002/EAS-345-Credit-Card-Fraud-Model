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
	src/
		01_preprocess.R
	    02_train_model.R
	    03_test_model.R
	README.md

How to Run the Project

	1. In R Studio, set your working directory to the project root:

	2. Preprocess the data:

		Run src/01_preprocess.R in R Studio

	3. Train the model

		Run src/02_train_model.R in R Studio

			Need to uncomment these two lines 

				#dir.create("models", showWarnings = FALSE)
				#xgb.save(xgb_model, "models/xgb_fraud.model")

			in order for the code to create the models directory and save the trained model to that directory.

			NOTE: the models directory already exists on the public repository, so do not uncomment that line of code and commit any changes. Best to leave those lines commented out, the model will run fine.				
	4. Test/evaluate the model

		Run src/03_test_model.R in R Studio

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