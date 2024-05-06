# House Price Prediction Project

## Overview

This project aims to develop machine learning models for predicting house prices based on various features related to the properties. The dataset used in this project contains comprehensive information about properties, including market value, construction details, building characteristics, and geographic information.

## Table of Contents

1. [Data Collection](#data-collection)
2. [Data Processing](#data-processing)
3. [Feature Engineering](#feature-engineering)
4. [Model Development](#model-development)
5. [Test Dataset Extraction](#test-dataset-extraction)
6. [Model Evaluation](#model-evaluation)
7. [Results](#results)
8. [Discussion](#discussion)

## Data Collection

The project utilizes a dataset containing information about properties, likely obtained from a real estate or municipal database. The dataset is quite comprehensive, with over 80 columns covering a wide range of features related to the properties, including market value, construction details (number of bedrooms, bathrooms, stories, basement), building characteristics (zoning, construction type, exterior condition), and geographic information (location, zip code).

## Data Processing

In the data preprocessing step, a significant number of columns (over 40) are dropped from the original dataset. These columns are likely deemed irrelevant or redundant for the specific task of predicting total livable area. Additionally, missing values in the critical 'market_value' and 'total_area' columns are addressed by imputing the mean values of those features.

## Feature Engineering

To enhance the predictive power of the models, feature engineering techniques are employed. Specifically, the number of bathrooms is incorporated as an additional feature, as it is recognized as a significant factor influencing house prices. The bathroom feature is engineered by replacing NaN values and integrating it into the dataset.

## Model Development

The project explores multiple regression models, including Linear Regression, Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor. These models represent a diverse set of approaches, ranging from simple linear models to complex ensemble methods. The dataset is appropriately split into training and test sets to ensure unbiased evaluation of the models' predictive performance.

## Test Dataset Extraction

A representative test dataset is extracted from the original data, following best practices in machine learning. This involves determining an appropriate split ratio (e.g., 80/20), employing stratified sampling techniques to ensure representativeness, and proper randomization to mitigate potential biases. The test set remains unseen by the models during the training process, ensuring an unbiased estimate of their generalization performance.

## Model Evaluation

The models are evaluated on the test set using a comprehensive set of regression metrics, including Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared. Additionally, the continuous predictions are converted to binary values, allowing for the calculation of classification metrics such as precision, recall, F1-score, and accuracy.

## Results

- **Linear Regression**: This model proved to be the most effective, with an impressive R-Squared value of 0.95, indicating that it explains 95% of the variance in the target variable. Additionally, the analysis suggests a positive correlation between house area and price, where larger houses generally have higher prices.

- **Decision Tree Regressor**, **Random Forest Regressor**, and **Gradient Boosting Regressor**: These models showed signs of overfitting, with an F1 value of 1.00, which is an indication of potential overfitting.

## Discussion

While the Linear Regression model demonstrated strong predictive performance, the ensemble models (Decision Tree, Random Forest, and Gradient Boosting) exhibited overfitting issues. Further refinements in feature engineering, model selection, and hyperparameter tuning may be necessary to enhance the ensemble models' performance and mitigate overfitting. Additionally, the report indicates that the R-squared values for the ensemble models are not satisfactory, suggesting that their predictive ability has room for improvement.