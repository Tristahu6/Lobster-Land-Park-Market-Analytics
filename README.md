# Market Analytics Project
# Lobster Land Data Analysis and Forecasting

## Project Background

Lobster Land, a theme park in Maine, is planning a grand celebration for the July 4th Independence Day holiday. The park is set to host an international conference of hoteliers and theme park operators from July 1st to July 4th. 

## Objective

The objective of this project is to analyze various aspects of the skiing-themed hotel industry to assist Lobster Land make proper business strategic decisions and conference discussion.

## Method 
To achieve our objective, we have segmented the project into several key steps:

1. **Data Collection**: We start by collecting data from the skiing-themed hotel industry, specifically from a dataset named "ski_hotels.csv."

2. **Data Cleaning and Preprocessing**: In this step, we clean and preprocess the data by removing unnecessary columns, adjusting column names, and handling missing values.

3. **Standardization (for some parts)**: We standardize the values of different variables to ensure they have the same scale for analysis.

4. **Segmentation**: Using K-Means clustering, we determine the optimal number of clusters and group skiing-themed hotels into different clusters based on various attributes.

5. **Conjoint Analysis**: Conjoint analysis is performed using customer survey data to understand the preferences of hotel guests regarding various hotel facilities. This analysis helps us identify the features and combinations that guests might prefer.

6. **Forecasting**: We forecast the annual net income for a selected hotel using time series forecasting techniques. First, we preprocess and analyze the financial data for the hotel. Then, we apply the ARIMA model to make revenue forecasts for the upcoming years.

7. **Classification**: We build a logistic regression model to predict guest satisfaction based on various factors. We assess the model's performance and provide recommendations for improving guest satisfaction.

8. **A/B Testing & Statistical Testing**: Lobster Land purchased three promotional photos and collected a dataset testing their popularity among a group. We've analyzed the data from these tests to assist Lobster Land in selecting one image for the conference invitations.
