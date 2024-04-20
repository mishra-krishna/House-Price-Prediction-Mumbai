# Mumbai Real Estate Price Prediction System

## Overview
The Mumbai Real Estate Price Prediction System is a web-based application designed to estimate property prices in Mumbai's housing market. Leveraging machine learning algorithms and user-provided property features, it offers predictions to assist potential buyers, sellers, and real estate professionals. This project is intended for skillset display and educational purposes only. Please note that the accuracy of the predictions may be limited due to data quality issues inherent to real-world datasets. Additionally, while the system uses a single dataset for prediction across multiple algorithms for comparison purposes, it is recommended to refine the dataset according to the specific requirements of each algorithm for optimal performance.

## Features
- **Data Preprocessing:** Clean and preprocess raw real estate data to prepare it for machine learning models.
- **Machine Learning Models:** Implement various regression algorithms, including Linear Regression, Decision Tree, Random Forest, and Support Vector Machine (SVM), to predict property prices.
- **Web Application:** Utilize Flask, a Python micro-framework, to develop a user-friendly web interface for interacting with the prediction system.
- **User Interaction:** Allow users to input property features such as area, number of bedrooms, bathrooms, and furnished status, and receive estimated property prices.
- **Model Evaluation:** Assess the performance of the machine learning models using evaluation metrics such as R-squared score to measure predictive accuracy.

## Usage
1. **Installation:** Install the required dependencies by running `pip install -r requirements.txt`.
2. **Run the Application:** Start the Flask web application by executing `python app.py`.
3. **Access the Application:** Navigate to the provided URL in your web browser to access the Mumbai Real Estate Price Prediction System.

## Data
The project utilizes a dataset containing real estate information specific to Mumbai. The dataset includes features such as property area, number of bedrooms and bathrooms, furnished status, and target variable (property price). Data preprocessing techniques are applied to handle missing values, categorical variables, and outliers, ensuring the quality of input data for the machine learning models.

## Credits
- **scikit-learn:** Python library for machine learning tasks
- **Flask:** Python web framework for building web applications
- **Bootstrap:** Front-end framework for developing responsive and mobile-friendly web interfaces


