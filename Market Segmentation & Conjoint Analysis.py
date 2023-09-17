#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing
# Import the dataset 
food_trucks=pd.read_csv('food_trucks.csv')
food_trucks.head()
# Drop the vendorID variable
df = food_trucks.drop('vendorID',axis=1)
# Check missing values
missing_values = df.isnull().sum()
print(missing_values)

##Segmentation/Clustering
from sklearn.preprocessing import StandardScaler
# Standardize the variables in the DataFrame
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(clean_df), columns=clean_df.columns)

# Create a new dataframe with 5 selected variables
df_zscore = df_standardized[['avg_transaction_cost', 'mnths_operational', 'days_yr', 'dist_lobland', 'bev_percent']].copy()
df_zscore

# Create Elbow chart
sse = {}
for k in range (1,10):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df_zscore)
    sse[k]=kmeans.inertia_
plt.xlabel('Number of Clusters (k)')
plt.ylabel('SSE')
plt.title('Elbow Method')
sns.pointplot(x=list(sse.keys()),y=list(sse.values()));

# Silhouette Analysis to determine a proper k
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df_zscore)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(df_zscore, labels))

plt.plot(range(2, 7), silhouette_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.show()

# Build a k-means model with 4 clusters
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(df_zscore)

# Retrieve the cluster labels for each data point
df_zscore['cluster_labels'] = kmeans.labels_

# Print the DataFrame with cluster labels
print(df_zscore)

# Check on the distribution of clusters
df_zscore['cluster_labels'].value_counts()

# Create a new DataFrame with original data and with cluster labels assigned
df_with_clusters = clean_df.copy()
df_with_clusters['cluster_labels'] = kmeans.labels_
print(df_with_clusters)

# Summary statistics about each cluster
grouped_df = df_with_clusters.groupby(['cluster_labels'])
for name, group in grouped_df:
    print(f"\nCluster {name}:")
    print(group.describe())
    
# Cluster visualization 1
sns.scatterplot(x='dist_lobland', y='mnths_operational', hue='cluster_labels', data=df_with_clusters)
plt.xlabel('dist_lobland')
plt.ylabel('mnths_operational')
plt.title('Clusters based on dist_lobland and mnths_operational')
plt.show()

# Cluster visualization 2
summary = df_zscore.groupby(['cluster_labels']).agg({
    'avg_transaction_cost': 'mean', 
'mnths_operational': 'mean',
'days_yr': 'mean',
'dist_lobland': 'mean',
'bev_percent': 'mean'}).round(2)

plt.figure(figsize=(9,4))
sns.heatmap(summary, annot=True, cmap='BuPu', fmt='g');

##Conjoint Analysis
woodie = pd.read_csv("woodie.csv") 
woodie.isnull().values.any()
#Dummy all variables
woodie3 = pd.get_dummies(woodie, drop_first=True, columns=['start_high', 'maxspeed', 'steepest_angle', 'seats_car', 'drop',
       'track_color'])
#Run linear regression
import statsmodels.api as sm
X = woodie3[['start_high_Yes', 'maxspeed_60', 'maxspeed_80', 'steepest_angle_75', 'seats_car_4', 'drop_200', 'drop_300', 'track_color_green', 'track_color_red', 'track_color_white']]
y = woodie3['avg_rating']
X=sm.add_constant(X)
model = sm.OLS(y, X).fit()
model.summary()

