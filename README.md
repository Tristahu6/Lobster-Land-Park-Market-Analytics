# Lobster Land Park Market Analytics
## Project Background

Lobster Land, a theme park in Maine, is planning a grand celebration for the July 4th Independence Day holiday. The park is set to host an international conference of hoteliers and theme park operators from July 1st to July 4th. 

## Objective

The objective of this project is to analyze various aspects of the skiing-themed hotel industry to assist Lobster Land make proper business strategic decisions and conference discussion.

## Method 
To achieve our objective, we have segmented the project into several key steps:

1. **Data Collection**: We start by collecting data from the skiing-themed hotel industry, specifically from a dataset named "ski_hotels.csv."

2. **Data Cleaning and Preprocessing**: In this step, we clean and preprocess the data by removing unnecessary columns, adjusting column names, and handling missing values.

3. **Segmentation**: Using K-Means clustering, we determine the optimal number of clusters and group skiing-themed hotels into different clusters based on various attributes.

4. **Conjoint Analysis**: Conjoint analysis is performed using customer survey data to understand the preferences of hotel guests regarding various hotel facilities. This analysis helps us identify the features and combinations that guests might prefer.

5. **Classification**: We build a logistic regression model to predict guest satisfaction based on various factors. We assess the model's performance and provide recommendations for improving guest satisfaction.

### 1. **Packages:**

#### 1.1 **`pandas` (alias `pd`):**
- Functions Used:
  - `pd.read_csv()`
  - `.head()`
  - `.info()`
  - `.describe()`
  - `.isnull().sum()`
  - `.drop()`
  - `.dropna()`
  - `.astype()`
  - `.groupby()`
  - `.agg()`
  - `.rename()`
  - `.replace()`

#### 1.2 **`numpy` (alias `np`):**
- Functions Used:
  - `np.nan`
  
#### 1.3 **`seaborn` (alias `sns`):**
- Functions Used:
  - `sns.pointplot()`
  - `sns.heatmap()`
  - `sns.countplot()`
  - `sns.scatterplot()`
  - `sns.despine()`

#### 1.4 **`matplotlib.pyplot` (alias `plt`):**
- Functions Used:
  - `plt.plot()`
  - `plt.xlabel()`
  - `plt.ylabel()`
  - `plt.title()`
  - `plt.xticks()`
  - `plt.scatter()`
  - `plt.colorbar()`
  - `plt.figure()`
  - `plt.show()`

#### 1.5 **`scikit-learn` (imported various modules):**
- Functions and Modules Used:
  - `KMeans`
  - `preprocessing`
  - `LabelEncoder`
  - `train_test_split`
  - `LogisticRegression`
  - `accuracy_score`
  - `confusion_matrix`
  - `classification_report`
  
#### 1.6 **`statsmodels.api` (alias `sm`):**
- Functions Used:
  - `sm.Logit()`
  - `result.fit()`

#### 1.7 **`matplotlib.patches` (alias `mpatches`):**

### 2. **Other Python Functions:**
- `print()`
- `.value_counts()`
- `.pivot_table()`
- `.sort_values()`
- `.round()`
