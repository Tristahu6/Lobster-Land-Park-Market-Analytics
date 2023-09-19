#!/usr/bin/env python
# coding: utf-8

# Project Background: Lobster Land - a fictional theme park, plans to host a big party to celebrate the July 4th Independence Day holiday. 
# The park will host an international conference of hoteliers and theme park operators from the 1st through the 4th of July. 
# The analysis include ... to prepare for this conference.

# Index: 
# Line 1-10 Exploratory 

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
import statsmodels.api as sm
import numpy as np
import seaborn as sns
from numpy.linalg import norm
import matplotlib.patches as mpatches
from sklearn.model_selection import GridSearchCV

#################### Exploratory data analysis of the park_accidents dataset ####################
park_accidents = pd.read_csv("park_accidents.csv")
park_accidents.head()
park_accidents.info()
park_accidents.describe()
park_accidents.isnull ().sum()

# Count of Accidents 
accidents = park_accidents.groupby('year')['acc_id'].count().describe()
print(accidents)

# Number of Accidents Per Year
yearly_accidents=park_accidents.groupby('year')["acc_id"].count()
yearly_accidents

# Visualization of accidents per year
plt.plot(yearly_accidents.index, yearly_accidents.values)
plt.xlabel('Year')
plt.ylabel('Accident Count')
plt.title('Accidents by Year')
plt.xticks(rotation=45)
plt.show()

# Count Accidents by Gender
accidents = park_accidents.groupby('gender')['acc_id'].count()

total_accidents = park_accidents['acc_id'].count()
monthly_accidents_proportion = park_accidents.groupby('gender')['acc_id'].count() / total_accidents
print(monthly_accidents_proportion)

# Accident Causes by Type
accident_causes = park_accidents.drop(['year', 'acc_id', 'age_youngest'], axis=1).groupby(['mechanical', 'employee', 'op_error']).sum()
print(accident_causes)

# Create a pivot table
pivot_table = park_accidents.pivot_table(index='category', values=['num_injured'], aggfunc='sum').sort_values(by='num_injured', ascending=False)
print(pivot_table)

# Based on the dataset, which contains 14,884 accident records. On average, each accident resulted in 1.08 injuries, 
# with the youngest individual involved being 0 years old and the oldest being 110 years old. 
# The number of accidents increased gradually from 1986 to 1999, with a significant rise observed between 1999 and 2002. 
# After reaching a peak in 2002 with 2,217 accidents, the numbers declined steadily until 2009, where only two accidents were recorded. 
# This indicates that the safety measures may be put in place in previous years and are effective. 
# Therefore, the park owner could consider retaining these safety measures. The dataset includes information on the gender of the individuals involved in accidents. Among the recorded cases, approximately 47% were female, 35% were male, and 17% had an unspecified gender. It seems like more female were involved and injuried (though this could because more female came to the park). The dataset provides insights into the causes of accidents. The most frequent accident cause was "Impact: hit something in participatory attraction" with 2,021 injuries, followed by "Load/Unload: scrape or stumble" with 1,507 injuries. However, it's important to note that the dataset has some limitations. The data failed to categorize the injury types (mechanical, employee, or op_error) for most of the cases, meaning the categories were not properly defined or the data collection were bad. Also, it does not provide detailed information about the specific attractions or parks where the accidents occurred, making it challenging to pinpoint exact locations or specific ride-related factors contributing to accidents. Additionally, the dataset does not include information on the severity of injuries or any long-term consequences. To gain a comprehensive understanding of the safety landscape, it would be beneficial to collect more detailed data on individual ride characteristics, thorough incident reports, and further demographics to analyze accident patterns based on factors such as age groups or visitor types.


######################################## Segmentation ########################################
# Analyze skiing-themed hotels in the industry
skihotels = pd.read_csv ("ski_hotels.csv")
skihotels.head()
skihotels.describe()
# removed column "unnamed" as it's meaningless
skihotels2 = skihotels.drop('Unnamed: 0', 1)
skihotels2.head()

# Adjust column names 
new_column_names = {
    'price (£)': 'price',
    'altitude (m)': 'altitude_m',
    'totalPiste (km)': 'total_piste_km',
    'totalLifts': 'total_lifts',
    'gondolas': 'num_gondolas',
    'chairlifts': 'num_chairlifts',
    'draglifts': 'num_draglifts',
    'blues': 'num_blues',
    'reds': 'num_reds',
    'blacks': 'num_blacks',
    'totalRuns': 'total_runs'}
skihotels2 = skihotels2.rename(columns=new_column_names)
skihotels2.isnull ().sum()
skihotels2 = skihotels2.replace('unknown', np.nan)

# Check NA values 
skihotels2.isnull ().sum()

# Due to the amount of NaN values for future analysis we will not be using the Distance from lift neither the snowfall columns. 
# Hoewever the sleeps column is relevant for our analisis therefore we will drop the 96 rows of values missing
skihotels2_new= skihotels2.dropna(subset=['sleeps'])

# Standarize the values of different variables as they have different units 
skihotels3 = skihotels2_new[['price',
       'altitude_m', 'total_piste_km', 'total_lifts', 'num_gondolas',
       'num_chairlifts', 'num_draglifts', 'num_blues', 'num_reds',
       'num_blacks', 'total_runs', 'sleeps']]
skihotels3.head()

score = preprocessing.StandardScaler ()
SH3_standard = score.fit_transform(skihotels3)
SH3_standard = pd. DataFrame (SH3_standard)
SH3_standard.columns=skihotels3.columns

round(SH3_standard. describe(), 2)
SH3_standard.head(5)

# Select number of clusters 
sse = {}
for k in range(1, 15):
 kmeans = KMeans (n_clusters=k, n_init=10, random_state=654)
 kmeans.fit (SH3_standard)
 sse[k] = kmeans.inertia_
plt.title('The Elbow Method')
plt.xlabel('k')
plt.ylabel('SSE')
sns.pointplot(x=list(sse.keys()),y=list(sse.values()));

kmeans = KMeans(n_clusters=3,random_state = 654)
kmeans.fit (SH3_standard)
cluster_labels = kmeans.labels_
SH3_standard2 = SH3_standard.assign(Cluster = cluster_labels)
SH3_standard2.groupby (['Cluster']) .agg({'price': 'mean',
'sleeps': 'mean','altitude_m': 'mean','total_piste_km': 'mean',
'total_lifts':'mean', 'total_runs':'mean'}).round(2)

kmeans = KMeans(n_clusters=2,random_state = 654)
kmeans.fit (SH3_standard)
cluster_labels = kmeans.labels_
SH3_standard2 = SH3_standard.assign(Cluster = cluster_labels)
SH3_standard2.groupby (['Cluster']) .agg({'price': 'mean',
'sleeps': 'mean','altitude_m': 'mean','total_piste_km': 'mean',
'total_lifts':'mean', 'total_runs':'mean'}).round(2)

kmeans = KMeans(n_clusters=5,random_state = 654)
kmeans.fit (SH3_standard)
cluster_labels = kmeans.labels_
SH3_standard2 = SH3_standard.assign(Cluster = cluster_labels)
SH3_standard2.groupby (['Cluster']) .agg({'price': 'mean',
'sleeps': 'mean','altitude_m': 'mean','total_piste_km': 'mean',
'total_lifts':'mean', 'total_runs':'mean'}).round(2)

kmeans = KMeans(n_clusters=4,random_state = 654)
kmeans.fit (SH3_standard)
cluster_labels = kmeans.labels_
SH3_standard2 = SH3_standard.assign(Cluster = cluster_labels)
SH3_standard2.groupby (['Cluster']) .agg({'price': 'mean',
'sleeps': 'mean','altitude_m': 'mean','total_piste_km': 'mean',
'total_lifts':'mean', 'total_runs':'mean'}).round(2)

# After revising the different clustering as it recorded a significant difference between cluster 3 but by cluster 5 this change was minimal.

summary_stats=SH3_standard2.groupby (['Cluster']) .agg({'price': 'mean',
'sleeps': 'mean','altitude_m': 'mean','total_piste_km': 'mean',
'total_lifts':'mean', 'total_runs':'mean'}).round(2)

plt.scatter(SH3_standard2 ['price'], SH3_standard2 ['total_piste_km'], c=SH3_standard2 ['Cluster'],  cmap='viridis')
plt.xlabel('price')
plt.ylabel('total_piste_km')
cbar = plt.colorbar()
cbar.set_label('Cluster')
plt.show()

print(summary_stats)

SHsum = skihotels2_new.assign(Cluster = cluster_labels)
SHsum.dtypes


# To compare price with sleeeps we must change the type of "sleeps" from object to int64

SHsum['sleeps'] = SHsum['sleeps'].astype('int64')
SHsum.dtypes

discretionary1 = SHsum.groupby('Cluster')[['price','sleeps']].mean ()
discretionary1

# In this graphic, we can see a tendency that as price rises, the amount of rooms decrease. We say tendency as cluster 1 falls outside of this idea.

discretionary2 = SHsum.groupby('Cluster')[['altitude_m','total_piste_km']].mean ()
discretionary2

# This graphic was done as a way to understand the features of the hotel and to see if there ws a trend between the length of the track and altitude of the ski start. The trend here is not necesarily clear but it can be seen that the higher the altitude the longer the track length.

discretionary3 = SHsum.groupby('Cluster')[['price','total_runs']].mean ()
discretionary3

# This is a different kind of feautures measure to understand how many activities there are n the hospitality and whether or not price inceases. Here it is clear without a doubt that as price increases the number of runs also does.

discretionary4 = SHsum.groupby('Cluster')[['price','total_lifts']].mean ()
discretionary4

# This follows the same idea as the previous graphic but we replaced total_runs with total_lifts. However the result is the same, the higher the price the larger the number of accessible lifts.

# We used the total number of lifts and the total number of runs to obtain a clearer clustering instead of using 6 more variables that basically will give the same result.

plt.figure(figsize=(10, 6))
plt.xlabel("price")
plt.ylabel("sleeps")
sns.despine()

sns.scatterplot(data = discretionary1, x='price', y='sleeps', hue="Cluster", palette="icefire")
plt.title("Price per number of rooms")
plt.xlabel("Price")
plt.ylabel("Sleeps")
plt.legend(title="Cluster")
plt.show()

# This graph basically just emphasizes the impact of cluster 0 in mantaining the trend where as rooms available decrease the price increases

summary = SHsum.groupby (['Cluster']) .agg({'price': 'mean',
'sleeps': 'mean','altitude_m': 'mean','total_piste_km': 'mean',
'total_lifts':'mean', 'total_runs':'mean'}).round (2)
plt.figure(figsize=(11,5))
sns.heatmap(summary, annot=True, cmap='BuPu', fmt='g');

# For this graph we will explain it by cluster. Price seems to be the lowest in cluster 2 which also contains the most rooms available, the shortest track, lower number of lifts and runs. However something that stands out is the altitude of the ski mountain not being the shortest as it goes against the idea that the lower the price the less amenities are present. If we see Cluster 3 which has the highest price, it i the second largest in most of the variables yet the altitude is the lowest. We can assume that customers will go there due to the exclusivity of it as it the smallest hotel room wise.

plt.figure(figsize=(22,10))
sns.countplot (x='total_piste_km', hue='Cluster', data=SHsum);

# This graph might seem all over the place but this is due to the amount of different track lengths, 
# the important thing here is that there is a clear clustering with cluster 2 having the shortest tracks and cluster 0 having the longest ones which go hand in hand with the amount of activities that can be done and the necessity of lifts.

plt.figure(figsize=(10, 6))
plt.xlabel("price")
plt.ylabel("sleeps")
sns.despine()

sns.scatterplot(data = SHsum, x='total_runs', y='total_piste_km', hue="Cluster", palette="icefire")
plt.title("Total runs per Total Piste in km")
plt.xlabel("Runs")
plt.ylabel("Piste")
plt.legend(title="Cluster")
plt.show()

# This graph just shows with clear clustering, how the number of runs increase as the price increments aprt from certain outliers. 
# we do however notice the muddling between cluster 0 and 3 which are in the same area and have harder time differentiating their trend.

SHsum['Cluster'] = SHsum['Cluster'].astype('category')
SHsum['Cluster'] = SHsum['Cluster'].cat.rename_categories({0:'Cash Cows', 1: 'Smooth Operators',
2:'Lower Elevation Resorts with Limited Facilities', 3: 'Extensive Ski Areas with Mid-range Pricing'})
SHsum['Cluster'] = SHsum['Cluster'].cat.remove_unused_categories()
SHsum['Cluster'].dtype


summary = SHsum.groupby (['Cluster']) .agg({'price': 'mean',
'sleeps': 'mean','altitude_m': 'mean','total_piste_km': 'mean',
'total_lifts':'mean', 'total_runs':'mean'}).round (2)
plt.figure(figsize=(11,5))
sns.heatmap(summary, annot=True, cmap='BuPu', fmt='g');


# Cluster 0: Premium Alpine Resorts Explanation: This cluster represents high-end ski resorts located at high altitudes with extensive ski slopes. These resorts 
# also offer a wide range of amenities. The higher prices and larger number of lifts and runs indicate a nicher and premium aesthetic. Customers could prefer this 
# place due to its exclusivity.
# Cluster 1: Affordable Mountain Retreats Explanation: This cluster represents moderately priced ski resorts that have a balanced combination of 
# features and accommodations. These resorts cater to budget-conscious skiers seeking a cost-effective holiday.
# Cluster 2: Down Under Explanation: This cluster represents ski resorts situated at lower altitudes with smaller and fewer ski slopes and lift facilities. 
# These resorts could offer a skiing experiencefor those beginners. Accommodations and features might be more basic.
# Cluster 3: Close to Premium, less exclusivity Explanation: This cluster is really close to cluster 1 with only the price being higher. 
# This represents ski resorts with a large number of runs and lifts but at a higher cost and with a lowest altitude of all. 
# These resorts offer a wide range of skiing opportunities which makes it attractive for high class customers that arent professionals as the cluster contains 
# good levels of slopes, and lifts.

######################################## Conjoint Analysis ########################################
# Conjoint analysis on customer survey data in terms of hotel facility to understand the features/combos that hotel guests might prefer

amenities=pd.read_csv("hotel_amenities.csv")
#suppressing scientific notation, two decimal places
pd.options.display.float_format = '{:.2f}'.format
amenities.head()
amenities.info()
amenities.isna().sum()

costs=pd.read_csv("amenity_costs.csv")

#dummify all except outcome variable
amenitiesdummy = pd.get_dummies(amenities, drop_first=True, columns= ['WiFi_Network','breakfast','parking','gym','flex_check','shuttle_bus','air_pure','jacuzzi','VIP_shop','pool_temp'])

amenitiesdummy.sort_values(by='avg_rating', ascending=False).head()

X = amenitiesdummy[['WiFi_Network_Best in Class', 'WiFi_Network_Strong',
       'breakfast_Full Buffet', 'breakfast_None', 'parking_Valet', 'gym_Basic',
       'gym_None', 'gym_Super', 'flex_check_Yes', 'shuttle_bus_Yes',
       'air_pure_Yes', 'jacuzzi_Yes', 'VIP_shop_Yes', 'pool_temp_80',
       'pool_temp_84']]
y = amenitiesdummy['avg_rating']

# build linear model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X,y)
regressor.intercept_

#display coefficients
coef_df=pd.DataFrame(regressor.coef_,X.columns,columns=['Coefficient'])
coef_df

# Notes: 
# Wifi: Basic was dropped; both others were more popular, Best in Class was most popular.
# Breakfast: Continental was dropped, None is less popular and Full Buffet is more popular.
# Parking: Open Lot dropped, Valet is very very slightly more popular (but a lot more expensive so maybe not worthwhile)
# Gym: Advanced was dropped, None and Basic were less popular and Super was more popular by a tiny margin.
# Flexible Check in : No dropped, Yes more popular
# Shuttle Bus: No dropped, Yes more popular
# Air Pure: No dropped, Yes very marginally more popular
# Jacuzzi: No dropped, Yes more popular
# VIP shop: No dropped, Yes more popular
# Pool temp: 76 dropped, 84 more popular, 80 very marginally more popular. Things to consider: is 84 realistic for a pool? Are people sure they know what they're asking for/opting for? It might get too warm and 80 may honestly be more than enough.

amenities2=amenitiesdummy.sort_values(by=['avg_rating'],ascending=False)
amenities2.head()

# Some overarching characteristics of the best rated packages: Strong WiFi network, full buffet breakfast, valet parking, Super gym, flexible check in, 
# shuttle bus, air purifier, no jacuzzi necessary, yes to a VIP shop, pool temperature at 76 or 80* F. 
# Based on the analysis, and also consider the budgeting for a strong wifi network, full buffet breakfast, no valet, advanced gym, flexible check in, shuttle bus, 
# no air purifier, no jacuzzi, yes VIP shop, and a pool temperature of 80*, 16.25+22.45+15+35+12+75+12+35 = we get a total cost of 222.7 per room/customer. 
# Given our budget of 250, we are left with 27.3. Hoteliers might choose to throw in air purifiers since it is not that much more to the costs (12.85). 
# In conclusion, our recommended combo is: a strong wifi network, full buffet breakfast, no valet, advanced gym, flexible check in, shuttle bus, add air purifier, no jacuzzi, yes VIP shop, and a pool temperature of 80*. 

######################################## Forecasting Revenue for selected hotel ########################################
hlt = pd.read_csv("HLT_annual_financials.csv")
hlt.info()
hlt.head()

# Data pre-processing
# Extract net income row
net_income_hlt = hlt[hlt['name'] == '\tNetIncome']
# Drop name and ttm columns
net_income_hlt = net_income_hlt.drop(columns=['name', 'ttm'])
# Transpose the DataFrame and convert index to datetime
net_income_hlt = net_income_hlt.transpose()
net_income_hlt.index = pd.to_datetime(net_income_hlt.index)
# Convert values to numeric, replacing any non-numeric characters
net_income_hlt = net_income_hlt.replace('[\$,]', '', regex=True).astype(float)
# Rename the column to 'Net Income'
net_income_hlt.columns = ['Net Income']
net_income_hlt.rename_axis('date', axis='index', inplace=True)
net_income_hlt.sort_index(inplace=True)

# Convert the net income values to millions
net_income_hlt['Net Income'] = net_income_hlt['Net Income'].apply(lambda x:round(x / 1e6, 2))
# Rename the column to 'Net Income (Millions)'
net_income_hlt.columns = ['Net Income in Millions']
net_income_hlt

missing_values_hlt = net_income_hlt.isna().sum()
missing_values_hlt


######################################## ARIMA model Forecasting Revenue for selected hotel ########################################
from statsmodels.tsa.stattools import adfuller
# Perform Augmented Dickey-Fuller test to check stationarity
ADFresult = adfuller(net_income_hlt['Net Income in Millions'])
print('ADF Statistic: %f' % ADFresult[0])
print('p-value: %f' % ADFresult[1])

# Plot the time series
plt.plot(net_income_hlt['Net Income in Millions'])
plt.title("Hyatt Annual Net Income")
plt.xlabel("Year")
plt.ylabel("Net Income in Millions")
plt.show()

# Differencing to make the series stationary
diff_net_income_hlt = net_income_hlt['Net Income in Millions'].diff().dropna()
# Check stationarity again
result = adfuller(diff_net_income_hlt)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
# Plot the differenced time series
plt.plot(diff_net_income_hlt)
plt.title("Differenced H Net Income in Millions")
plt.show()


# The ADF statistic is -2.575224, and the corresponding p-value is 0.098227. In this case, the p-value is greater than the commonly used significance level of 0.05. 
# This suggests that we fail to reject the null hypothesis, which means the differenced series is not stationary.

# Differencing again to make the series stationary
diff2_net_income_hlt = diff_net_income_hlt.diff().dropna()
# Check stationarity again
result2 = adfuller(diff2_net_income_hlt)
print('ADF Statistic: %f' % result2[0])
print('p-value: %f' % result2[1])
# Plot the differenced time series
plt.plot(diff2_net_income_hlt)
plt.title("Differenced H Net Income (Millions)2")
plt.show()

from statsmodels.graphics.tsaplots import plot_acf

# Extract the one-dimensional time series
time_series_hlt = net_income_hlt['Net Income in Millions'].values

# Plot the ACF of the time series
plot_acf(time_series_hlt, lags=10)
plt.show()

# ACF plot shows a significant spike at lag 0, it suggests that the autocorrelation at lag 1 is strong. In this scenario, we will start with an autoregressive order parameter 'p' of 1 for ARIMA

import statsmodels.api as sm

# Create the ARMA model
model1 = sm.tsa.ARIMA(diff_net_income_hlt, order=(1, 2, 1))

# Fit the model to the data
model_fit1 = model1.fit()

# Print the AIC value
print("AIC:", model_fit1.aic)

# Predict the values using the fitted model
time_series = net_income_hlt['Net Income in Millions']
predictions1 = model_fit1.predict(start=0, end=len(time_series)-1)

# Plot the predicted values
plt.figure(figsize=(11, 6))
plt.plot(net_income_hlt.index, net_income_hlt['Net Income in Millions'], label='Original Data')
plt.plot(net_income_hlt.index, predictions1, label='Predicted Data');


######################### Simple Exponential Smoothing (SES) model model Forecasting Revenue for selected hotel ########################################

# Since the time series dataset shows neither trend nor seasonality, so we also explored SES model for the forecasting

from statsmodels.tsa.api import SimpleExpSmoothing

time_series = net_income_hlt['Net Income in Millions']

# Create and fit the SES model with adjusted alpha
model2 = SimpleExpSmoothing(time_series)
model_fit2 = model2.fit(smoothing_level=0.3)

# Predict the values using the fitted model
predictions2 = model_fit2.predict(start=0, end=len(time_series)-1)
print(predictions2)


import matplotlib.pyplot as plt

# Plot the original data
plt.figure(figsize=(11, 6))
plt.plot(net_income_hlt.index, net_income_hlt['Net Income in Millions'], label='Original Data')

# Plot the predicted values
plt.plot(net_income_hlt.index, predictions2, label='Predicted Data')

# Set labels and title
plt.xlabel('Date')
plt.ylabel('Net Income (Millions)')
plt.title('Net Income (Millions) - Original vs Predicted')

# Add legend
plt.legend()

# Show the plot
plt.show()


# In[83]:


# Print the AIC value
print("AIC:", model_fit2.aic)


# ARIMA AIC: 174.64138911098465
# SES AIC: 170.7217375948289
# Since SES model has a smaller AIC, we'll use model SES to do the net income forecast for 2023

from statsmodels.tsa.api import SimpleExpSmoothing
import pandas as pd

# Extend the time index for predictions
last_date = time_series.index[-1]
next_year = last_date + pd.DateOffset(years=1)
index_extended = pd.date_range(start=last_date, end=next_year, freq='A')


# Predict the values for the extended time index
predictions_2023 = model_fit2.predict(start=last_date, end=index_extended[-1])

# Print the predictions for the next year
print(predictions_2023)


######################### Classification ########################################

hotel_satisfaction_raw = pd.read_csv('hotel_satisfaction.csv')
hotel_satisfaction_raw.head()

hotel_satisfaction = hotel_satisfaction_raw.drop("id", axis=1)

missing_values = hotel_satisfaction.isna().sum()
print(missing_values)

value_counts_satis = hotel_satisfaction['satisfaction'].value_counts()
print(value_counts_satis)

hotel_satisfaction.describe()


# Per checked, no outliers, no missing value, no impossible value and no class imbalance in outcome variable "satisfaction"

hotel_satisfaction.info()

# Build a correlation matrix
selected_vars = ['Age','Hotel wifi service', 'Departure/Arrival  convenience', 'Ease of Online booking',
       'Hotel location', 'Food and drink', 'Stay comfort',
       'Common Room entertainment', 'Checkin/Checkout service',
       'Other service', 'Cleanliness']
subset_df = hotel_satisfaction[selected_vars]
correlation_matrix = subset_df.corr().round(2)
# Create a heatmap of the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# A correlation of 0.72 between Hotel wifi service and Ease of Online booking is relatively high,
# but not high enough to present a significant problem with multicollinearity. Therefore, it is not
# necessary to remove any variables.

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(hotel_satisfaction['satisfaction'])
hotel_satisfaction['satisfaction'] = label_encoder.transform(hotel_satisfaction['satisfaction'])

# Check the updated DataFrame
print(hotel_satisfaction['satisfaction'])

# Dummify categorical variables
satisfaction_dummies = pd.get_dummies(hotel_satisfaction, columns=['Gender', 'purpose_of_travel', 'Type of Travel', 'Type Of Booking'], drop_first=True)
satisfaction_dummies

satisfaction_dummies.columns


# Create a data partition, make 40%/60% split for test and training sets
X = satisfaction_dummies[['Age', 'Hotel wifi service', 'Departure/Arrival  convenience',
       'Ease of Online booking', 'Hotel location', 'Food and drink',
       'Stay comfort', 'Common Room entertainment', 'Checkin/Checkout service',
       'Other service', 'Cleanliness', 'Gender_Male',
       'purpose_of_travel_aviation', 'purpose_of_travel_business',
       'purpose_of_travel_personal', 'purpose_of_travel_tourism',
       'Type of Travel_Personal Travel', 'Type Of Booking_Individual/Couple',
       'Type Of Booking_Not defined']]
y=satisfaction_dummies[['satisfaction']]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=829)

print(X_train.shape, X_test.shape, type(X_train), type(X_test))
print(y_train.shape, y_test.shape, type(y_train), type(y_test))

# Run logistic regression
logit_model1=sm.Logit(y_train, sm.add_constant(X_train))
result=logit_model1.fit()
print(result.summary())


# Numeric variables showing high p-values: none
# Categorical variables showing high p-values for ALL of the levels: Gender & purpose_of_travel -> whole variables to be removed

# Drop high pvalue variables and run logistic regression again
X_train2 = X_train.drop(['Gender_Male',
       'purpose_of_travel_aviation', 'purpose_of_travel_business',
       'purpose_of_travel_personal', 'purpose_of_travel_tourism'], axis=1)
logit_model2 = sm.Logit(y_train, sm.add_constant(X_train2))
result2 = logit_model2.fit()
print(result2.summary())

# Using scikit-learn
logmodel = LogisticRegression()
logmodel.fit(X_train2, y_train)
LogisticRegression()

# Make predictions and assess the performance against the train set
from sklearn.metrics import accuracy_score
predictions1 = logmodel.predict(X_train2)
accuracy = accuracy_score(y_train, predictions1)
print("Accuracy against the train set:", accuracy)


# Make predictions and assess the performance against the test set
X_test2 = X_test.drop(['Gender_Male',
       'purpose_of_travel_aviation', 'purpose_of_travel_business',
       'purpose_of_travel_personal', 'purpose_of_travel_tourism'], axis=1)
predictions2 = logmodel.predict(X_test2)
accuracy = accuracy_score(y_test, predictions2)
print("Accuracy against the test set:", accuracy)

# Calculate the accuracy and print the classification report
print(classification_report(y_test, predictions2))


# Build confusion matrix for test dataset
mat2=confusion_matrix(predictions2, y_test)
sns.heatmap(mat2, square=True, fmt = 'g', annot=True, cbar=False)
plt.xlabel('actual result')
plt.ylabel('predicted result')
a,b = plt.ylim()
a+=.5
b-=.5
plt.ylim(a,b)
plt.show()


# Sample prediction with Mary
# Create a DataFrame from the guest's information
Mary_df = pd.DataFrame({'Age': [30],
                        'Hotel wifi service': [3],
                        'Departure/Arrival  convenience': [2],
                        'Ease of Online booking': [4],
                        'Hotel location': [3],
                        'Food and drink': [4],
                        'Stay comfort': [5],
                        'Common Room entertainment': [3],
                        'Checkin/Checkout service': [4],
                        'Other service': [4],
                        'Cleanliness': [4],
                        'Type of Travel_Personal Travel': [1],
                        'Type Of Booking_Individual/Couple': [1],
                        'Type Of Booking_Not defined': [0]})

# Make a prediction using the logistic Regression model
prediction = logmodel.predict(Mary_df)

# Print the prediction
if prediction[0] == 1:
    print("The fictional guest is predicted to be satisfied with the hotel.")
else:
    print("The fictional guest is predicted to be neutral or dissatisfied with the hotel.")


# First, the logistic regression model provides valuable insights for business owners to efficiently improve guest satisfaction. The model reveals that factors such as hotel WiFi service and Common Room entertainment significantly impact guest satisfaction. This suggests that investing in reliable high-speed WiFi and enhancing room entertainment can positively influence guest experiences and satisfaction. To address these findings, the hotel should prioritize improving the quality and reliability of its WiFi service. This could involve investing in faster internet connections, ensuring good coverage throughout the hotel premises, and providing seamless connectivity for guests. Additionally, the hotel should pay attention to guests traveling for personal reasons or with an individual/couple booking type, as they may have lower satisfaction levels compared to other categories. By analyzing the specific needs and preferences of these guest segments, the hotel can tailor its services to meet their expectations. Offering personalized amenities, creating packages or experiences for couples or families, and understanding the unique requirements of personal travelers can help enhance their satisfaction levels.
# 
# Secondly, the logistic regression model can assist the hotel in predicting whether a customer will feel satisfied or not based on their profile - similar to the sample prediction case of Mary. This enables the hotel to focus more attention on specific individuals/groups predicted as neutral or dissatisfied and proactively prevent dissatisfaction. By identifying customers who are predicted to be unsatisfied and potentially unlikely to return to the hotel, the hotel can take pre-emptive measures such as offering rewards or membership programs during their check-in/check-out to mitigate the chances of dissatisfaction.
# 
# Thirdly, the hotel can design a tiered marketing promotional strategy to win back customers. By utilizing the predict_proba method in the logistic regression model, the hotel can determine the probabilities of satisfaction or dissatisfaction for individual customers. The hotel can then target a specific group of customers whose predict_proba values fall within a certain range, such as 0.40 to 0.60. This group, which comprises approximately 25 percent of the test set, may benefit from personalized outreach efforts. By sending promotional emails or messages to this group, the hotel can provide targeted incentives and information to regain their loyalty. Identifying and focusing on this specific group allows the hotel to implement effective marketing campaigns aimed at winning back these customers.
# 
# By implementing these recommendations, the hotel can enhance customer satisfaction, improve guest experiences, and potentially attract repeat customers and positive reviews, ultimately leading to long-term business success.

# # Strategic Memo

# In 2001 McDonald's corporation opened its first hotel called the Golden Arch Hotel, in the Swiss town of Rumlang. The hotel was a 4 star hotel with 211 beds. The hotel also opened a second location in Lully. The Golden Arches hotel was a Switzerland run McDonalds venture. This strategy took place only in this location The CEO Urs Hammer, perused this venture in 1999. The venture was approved by the board located in Chicago.  
# 
# The hotel project was invested with $32 million CHF and $26 Million USD, the Swiss subsidiary of McDonald’s was the ones who came up with the strategy for the Golden Arches hotel. It was to cater to middle class people. The hotel included a “170 seat drive through McDonalds restaurant that's open 24 hours a day (very unusual in Switzerland)”(Michel, 2005). The hotel offered two types of rooms, room 1 had a king size bed, and room two offered two oversized single beds. The price was set from $150 CHF to $200 CHF per night. The hotel offer ended a unique offering like self check in. 
# 
# Despite the technological advancements of having self check-in, Analysts were not convinced that this expansion fit well with McDonald's overall strategy. According to the article analysts claim the following. "I've just came back from lunch at McDonald's. But I can't imagine staying at a McDonald's hotel on a business trip," said Rene Weber at Bank Vontobel." Erwin Brunner, an asset manager at Brunner Invest AG, was more open-minded: "I usually stay in five-stars. But if there isn't one around, why not stay at McDonald's?" Peter Oakes, an industry expert at Merrill Lynch, was less optimistic, and "would be surprised if the Golden Arch Hotel expands to other countries." Robert LaFleur, an analyst with Bear Stearns in New York, noted that while McDonald's had a favorable brand image associated with conve-nience, hospitality, and cleanliness, he didn't expect the company to begin rapid expansion of hotels in the next few years. LaFleur described the Swiss venture as a blip on the radar screen for major U.S. hotel chains”(Michel, 2005) The industry experts lead us to believe that the idea of this hotel could be successful, however they do not see it expanding and may have mixed feelings and connotations about the hotel. 
# 
# In order for the hotel to be successful they need to be financially well off. And according to the profit indicated in the article the “volume of the building, the hotel part was nine times larger than the restaurant part, resulting in an investment of 28.8 million CHF ($23.15 million USD); 20% was equity, 80% was financed by a three-year mortgage for 3% per annum. The standard depreciation was 5% per annum of the nominal investment. The projected profit before taxes of 1,165,566 CHF ($936,950 USD) would lead to a 20.24% annual return of the 5.76 million CHF ($4.63 million USD) investment in a country with almost no inflation (see Exhibit 8 for a cost breakdown).”(Michel, 2005)
# 
# SWOT Analysis 
# 
# After reading the article, it is seen that the Golden Arches hotel had some strengths when entering the Swiss hotel market. The strengths were being technologically savvy and having self check-in. The hotel offered a unique self check-in feature, which could enhance convenience and efficiency for guests, especially those who were in a rush. The technological advancement in having a self serve option also shows that the hotel has a new fresh perspective. The second strength that is going to be touched on is the Investment in quality. The hotel invested in expensive mattresses, indicating a commitment to providing a comfortable sleeping experience for guests. Having comfortable beds also allow for customers to want to return to the hotel as that is one of the main aspects to a positive hotel room experience. 
# 
# Weaknesses:
# 
# However, a major weakness was the competition from nearby hotels. The presence of 17 new hotels nearby could lead to increased competition, potentially affecting occupancy rates and revenues. It could also affect the return rate, as people enjoy staying at new hotels. With that being said, the location challenge they face is that the hotel was situated in an area experiencing a large increase in hotels, specifically in the Zurich airport axis. This may have intensified competition and made it harder for the Golden Arch Hotel to stand out. The second weakness is staffing issues. The hotel overlooked staffing, and the Swiss human resource market was described as dried out. This could result in a lack of skilled or friendly staff, negatively impacting the guest experience. This could also affect the operations of the hotel running successfully. This weakness also was shown through customer experience reviews of having unfriendly staff: It is mentioned that the staff at the hotel were unfriendly, which could lead to a poor guest experience and negatively impact the hotel's reputation. Lastly, the article suggests that the rooms felt like an airport -like atmosphere. Some customers felt that the hotel resembled an airport experience, with noisy rooms and a sense of isolation. This could deter guests looking for a more relaxing and comfortable hotel environment.
# 
# Opportunities:
# 
# Opportunities for the Golden Arch hotel could be Improving guest experience. By addressing the weaknesses related to staff friendliness, room noise, and the airport-like atmosphere, the hotel can work towards providing a more pleasant and welcoming environment for guests. By ensuring a friendly environment it creates a warm and welcoming space for guests.
# 
# Another opportunity will be by targeting specific customer segments. The breakdown of guests by segment indicates an opportunity to focus on individual tourists who pay higher rates. Targeted marketing efforts towards this segment could help increase revenue per room. By targeting a specific demographic of staff, can also ensure that the amenities and services are catered towards their clientele. 
# 
# It is recommended that the Golden Arches Hotel, improve guest experiences, by hiring friendly staff, and training them to appropriately handle guest complaints and services. It is also important to redesign the rooms to feel more comfortable for the guests and so they feel less empty. They can also fill the hotel with more amenities to feel “more full”. Lastly, they should focus on targeting specific demographic segments and ensuring that they are catering to the correct audience. 
# 
# Threats:
# 
# The Golden Arches hotel is facing threats to the business. The first one being discussed is the intense competition. The presence of numerous hotels in the area, including 17 new ones, poses a threat to the Golden Arch Hotel's occupancy rates and average room rates. This is also impacting the return rate of guests. This is because there is high competition surrounding the hotel in terms of cost, design, and freshness. Another threat is the negative perceptions. The skepticism expressed by industry experts and the mixed feelings of analysts about the hotel could create negative perceptions among potential customers, affecting their decision to stay at the hotel. Another shift that can be a threat to the hotel is the change in customer preferences. Shifts in customer preferences and demands could impact the hotel's ability to attract and retain guests. It is important to monitor trends and adapt accordingly. With the hotel having a theme of the Golden Arches, customers may see this as a one time experience, or a family fun hotel. The theme of the hotel can also deter a certain demographic from staying at the hotel. When people vacation they may want to feel more luxurious and the theme can prevent people from feeling like this.  
# 
# Branding and positioning
# 
# The hotels, branding and positioning of the hotel was successful, as they made the look and feel of the hotel match the McDonald's brand. However, McDonald's overall strategy had left analysts expressing skepticism about the hotel's expansion, indicating that it may not align well with McDonald's overall brand and strategy. The branding and positioning of the hotel was done well, however they could have done a better job. The branding of the hotel could have been a bit more luxurious in the sense that it would have allowed them to cater towards a higher class of hotel goers. It could have also allowed for a higher revenue stream. The design of the hotel could also have been a little less like you are walking into a McDonald’s and more elevated. 
# 
# Conclusion 
# 
# The Golden Arches Hotel in Switzerland had a mix of strengths and weaknesses in its venture into the hotel market. While it showcased advanced technology and amenities, it faced challenges such as unfriendly staff, noise levels resembling an airport hotel, and difficulties in staffing. The analysts' opinions were divided, indicating limited potential for expansion into other markets. To succeed, the hotel needed to address its weaknesses, enhance the customer experience, and align its strategy with market demands. If they chose to re-enter the market it would be essential to seek consultants and advisors on more of the decisions being made in the strategy process within the hospitality industry. Similarly, Lobster Land can draw valuable insights from this case and apply them to their own business.
# 
# Firstly, Lobster Land should prioritize having friendly staff who provide excellent customer service. Unfriendly staff was identified as a weakness for the Golden Arches Hotel, and addressing this aspect can greatly enhance the overall guest experience at Lobster Land. Lobster is in the service industry just as the hotel, so this must be important for lobster as well. Secondly, Lobster Land should focus on continuously innovating and improving its offerings to enhance the customer experience. The Golden Arches Hotel showcased advanced technology and amenities, indicating the significance of staying updated with industry trends and adopting new innovations to meet customer expectations. Thirdly, Lobster Land should align its strategy with market demands. The Golden Arches Hotel faced challenges with noise levels and staffing difficulties, which impacted its competitiveness. By understanding and adapting to the market's demands, Lobster Land can ensure that its offerings meet the preferences and needs of its target audience. Furthermore, Lobster Land can benefit from seeking consultants and advisors within the hospitality industry. The Golden Arches Hotel's analysts had divided opinions, highlighting the importance of seeking expert guidance in decision-making processes. Lobster Land can consider consulting industry professionals to gain insights, receive recommendations, and make informed strategic decisions. Lastly, Lobster Land should stay updated with market trends and competitive pricing. By monitoring market dynamics, Lobster Land can adjust its offerings, pricing strategies, and marketing initiatives to remain competitive and attract customers.
# 
# Incorporating these recommendations into Lobster Land's strategy can help address weaknesses, enhance the customer experience, and align the business with market demands. This will increase the chances of success and improve Lobster Land's position in the industry.
# 

######################### Statistical Testing ########################################

promopics = pd.read_csv("promo_pics.csv")
promopics.head()
promopics.info()

plt.figure(figsize=(8, 6))
sns.barplot(x='pic_seen', y='site_duration', data=promopics, ci=None, color='red', alpha=0.7)
plt.xlabel('Picture Seen')
plt.ylabel('Site Duration')
plt.title('Average Site Duration for Each Picture')

summary_stats = promopics.groupby('pic_seen')['site_duration'].describe()
print(summary_stats)

# The bar chart above illustrates the average duration of site visits per 'pic seen'. It reveals that the Main St image captures the most user attention, as visitors spend the longest time viewing it. The site duration for the sunset picture is slightly lower than that of Main St, while the waterslide image has the shortest average duration. It is evident that the site duration varies across the different pictures, with 'Main St' having the highest mean duration, followed by 'Sunset', and then 'Waterslide'. Furthermore, the standard deviation values for each group indicate variability in site durations within their respective categories. Notably, the picture of Main St exhibits the greatest variability, suggesting a stronger preference among individuals who find it appealing.

print(promopics.isnull().sum())

import scipy.stats as stats
from scipy.stats import ttest_ind

# sunset 1 vs. Main st 2
t_statistic1, p_value = ttest_ind(promopics[promopics['pic_seen'] == 'Sunset']['site_duration'], promopics[promopics['pic_seen'] == 'Main St']['site_duration'])
print("sunset 1 vs. Main st 2:")
print("t-statistic:", t_statistic1)
print("p-value:", p_value)

# main st 2 vs. waterslide 3
t_statistic2, p_value = ttest_ind(promopics[promopics['pic_seen'] == 'Main St']['site_duration'], promopics[promopics['pic_seen'] == 'Waterslide']['site_duration'])
print("Main st 2 vs. waterslide 3:")
print("t-statistic:", t_statistic2)
print("p-value:", p_value)

# sunset 1 vs. waterslide 3
t_statistic3, p_value = ttest_ind(promopics[promopics['pic_seen'] == 'Sunset']['site_duration'], promopics[promopics['pic_seen'] == 'Waterslide']['site_duration'])
print("sunset 1 vs. waterslide 3:")
print("t-statistic:", t_statistic3)
print("p-value:", p_value)


# First, we created a bar chart that indicated Main St had the highest average views in terms of site duration. We also generated a summary statistics chart, which confirmed that Main St had the highest mean duration, followed by "Sunset," and then "Waterslide." Additionally, the standard deviation values for each group revealed variability in site durations within their respective categories. After ensuring that no NA values needed to be removed from the data, we conducted a two-sample T-test. The T-test results showed that the t-statistic for Sunset 1 vs. Main St 2 was -9.911415184705456, with a p-value of 1.0724418388111107e-22. For Main St 2 vs. Waterslide 3: the t-statistic is 112.12607762440105, and the p-value is 0.0, which is extremely small. This again suggests strong evidence against the null hypothesis, indicating a highly significant difference in site durations between Main St 2 and Waterslide 3, and between Sunset 1 vs. Main St 2. Based on the T-test results, we reject the null hypothesis for all comparisons. The extremely small p-values provide strong evidence against the null hypothesis, supporting the conclusion that Main St has a significantly higher site duration compared to both Sunset and Waterslide. Therefore, it is recommended that Lobster Land use the Main St image in their email campaign to maximize the view rate for the conference invites. 

# # Conclusions

# Based on our statistical analysis of the accident dataset, consisting of 14,884 records spanning from 1986 to 2009, we have obtained valuable insights to guide our decision-making and enhance the safety measures at our park. We observed a gradual increase in the number of accidents from 1986 to 1999, with a significant rise between 1999 and 2002. However, since then, there has been a steady decline in accidents, with only two recorded in 2009. This trend suggests that the safety measures implemented in previous years have been effective in reducing accidents. Thus, we recommend the following actions to the management team. Firstly, continue to prioritize safety as a core value and reinforce existing safety protocols and risk management strategies. Regular safety inspections and maintenance checks on all attractions are essential to ensure they meet the highest safety standards. . Furthermore, our analysis revealed insights into the causes of accidents. The most frequent accident cause was "Impact: hit something in participatory attraction," resulting in 2,021 injuries, followed by "Load/Unload: scrape or stumble" with 1,507 injuries. These findings emphasize the need for targeted measures to prevent collisions and enhance the safety of our participatory attractions. By addressing these specific causes, we can significantly reduce the risk of accidents and ensure a safer experience for our visitors. However, it's important to notice that these top injury cases doesn't seem like serious or damaging. The park should collect more data on the severity of the injury so that it can properly prioritize the problems to focus on and reduce the severe injuries that could keep customers away from the park and safety problems caused by the park itself. Also, implement targeted measures to prevent collisions and improve the safety of participatory attractions. This may involve redesigning ride layouts, implementing additional safety barriers, and enhancing staff training on crowd management. By implementing these recommendations and continuously monitoring and improving our safety practices, we can enhance the safety landscape of our park and ensure the well-being and satisfaction of our visitors. Let us remain committed to providing a safe and memorable experience for all our guests.
# 
# Secondly, the logistic regression model used in this analysis provides valuable insights for business owners to improve guest satisfaction efficiently. We identified the factors that significantly impact guest satisfaction, such as hotel WiFi service and Common Room entertainment. To address the findings, the hotel should prioritize enhancing the quality and reliability of its WiFi service. This can involve investing in faster internet connections, ensuring good coverage throughout the hotel premises, and providing seamless connectivity for guests. Additionally, the park should design a tiered marketing promotional strategy using the the probabilities of satisfaction or dissatisfaction predicted by the model, to win back customers. The hotel can target a specific group of customers who fall within a certain predict_proba range. These customers, who may benefit from personalized outreach efforts, can be reached through promotional emails or messages, providing targeted incentives and information to regain their loyalty.
# 
# Lastly, we recommend a combination of features for the hotel packages is as follows: a strong WiFi network, full buffet breakfast, no valet parking, an advanced gym, flexible check-in, a shuttle bus service, the option to add air purifiers, no jacuzzi, a VIP shop, and a pool temperature of 80°F. This combination provides a balance between customer preferences and cost-effectiveness. Based on the recommended combination, the estimated cost per room/customer is $235.55, which is slightly under the budget of $250. This allows for some flexibility and potential profit margin or a buffer to cover any losses from complimentary services provided to dissatisfied customers. In conclusion, the recommended package combination balances customer preferences, cost considerations, and revenue opportunities. By offering a strong WiFi network, full buffet breakfast, advanced gym, shuttle service, and other selected features, the hotel can provide a satisfying experience while maximizing cost-effectiveness.
