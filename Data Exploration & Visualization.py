##Data Exploration 
import pandas as pd
# Read the CSV file 
lobster22 = pd.read_csv('lobster22.csv')
lobster22.head(10)
lobster22.shape
lobster22.info()
lobster22['Precip'].value_counts()

#Data Cleansing - replace "T" with "0"
lobster22_2=lobster22
lobster22_2['Precip'] = lobster22_2['Precip'].replace("T", 0)

#Transferring data type to "float"
lobster22_2['Precip'] = lobster22_2['Precip'].astype(float)

#Data correction
lobster22_2.at[12, 'LowTemp'] = 51
lobster22_2.at[12, 'HighTemp'] = 74

#Some summary statistics
mean_LobsteramaRev = lobster22_2['LobsteramaRev'].mean()
std_LobsteramaRev = lobster22_2['LobsteramaRev'].std()
cv_LobsteramaRev = (std_LobsteramaRev / mean_LobsteramaRev) 
print(mean_LobsteramaRev, std_LobsteramaRev, cv_LobsteramaRev)

#Check missing values
lobster22_2.isna().sum()

#Renaming variable 'season_passholders'
lobster22_3 = lobster22_2.rename(columns={'season_passholders': 'SeasonPass'})

#Add one variable indicating weekday/weekend
recategorize_mapping = {'Monday': 'Weekday', 'Tuesday': 'Weekday', 'Wednesday': 'Weekday',
                        'Thursday': 'Weekday', 'Friday': 'Weekend', 'Saturday': 'Weekend', 'Sunday': 'Weekend'}
lobster22_3['DaySchedule'] = lobster22_3['Weekday'].map(recategorize_mapping)


##Data Visualization
#Histagram
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(x='MerchRev', hue='DaySchedule', multiple='stack', data=lobster22_3)
plt.title('Histogram of the Distribution of Merchandise Revenue per DaySchedule')
plt.show()

#Scatterplot
sns.scatterplot(x='TotalPax', y='CSD_Complaints', alpha = 0.4, data=lobster22_3)

import numpy as np
#Add a categorical variable rain intensity for the scatterplot  
lobster22_3['rain_intensity'] = np.where(lobster22_3['Precip'] > 0.3, 'big rain',
                                np.where(lobster22_3['Precip'] > 0, 'small rain', 'no rain'))
#Generate another scatter plot to see the GoldZone revenue on small-rainy vs. big-rainy vs. non-rainy days
sns.scatterplot(x='TotalPax', y='GoldZoneRev', hue='rain_intensity', data=lobster22_3)

#Faceted histograms - Ridership on rainy days vs. non-rain days
Faceted = sns.FacetGrid(data=lobster22_3, col='Weather')
Faceted.map(sns.histplot, 'Ridership', bins=10)
plt.show()
