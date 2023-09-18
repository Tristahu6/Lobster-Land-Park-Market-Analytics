#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 


# # Part I: Working with Time Series Data

# A. BeiGene, Ltd. (BGNE)

# In[2]:


# C-a
# Read the dataset 'BGNE.csv'
df = pd.read_csv('BGNE.csv', index_col='Date', parse_dates=True)

# Check the result
print(df.head())


# In[3]:


# C-b
df.info()


# D. 
# 
# Yes, this dataframe is indexed by time values. The Date column has been set as the index and was converted into a datetime data type using the "parse_dates=True" parameter. As indicated by the results of the info() function, the "Date" column is the index column of the DataFrame and has a "DatetimeIndex" type, ranging from 2022-06-17 to 2023-06-16.

# In[4]:


df.index


# In[5]:


# E-a.
max_value = df.index.max()
min_value = df.index.min()

print("Maximum value of the index: ", max_value)
print("Minimum value of the index: ", min_value)


# In[6]:


# E-b.
argmax_value = df.index.argmax()
argmin_value = df.index.argmin()

print("argmax value of the index: ", argmax_value)
print("argmin value of the index: ", argmin_value)


# E-c. 
# 
# The maximum value of the index (max) indicates the highest value present in the index column of the DataFrame. In this case, it is "2023-06-16 00:00:00," which is the latest date in the index.
# 
# The minimum value of the index (min) represents the lowest value present in the index column of the DataFrame. In this case, it is "2022-06-17 00:00:00," which is the earliest date in the index.
# 
# The argmax value of the index (argmax) refers to the position (or index location) of the maximum value in the index column. In this case, the value "250" indicates that the maximum value is located at index position 250.
# 
# The argmin value of the index (argmin) refers to the position (or index location) of the minimum value in the index column. In this case, the value "0" indicates that the minimum value is located at index position 0.

# In[7]:


# F-a
df.plot()
plt.show()


# F-a-i. 
# 
# The graph includes six different types of data: Open, High, Low, Close, Adjusted Close, and Volume. However, the graph only shows two types of data, which suggests that the other four types of data may not be visible due to differences in their ranges or magnitudes.
# 
# This graph presents challenges in interpretation for several reasons. Firstly, the different ranges or magnitudes of the variables can cause some data to be hidden or overshadowed by others. Variables with smaller units or lower volatility may not be easily distinguishable in the graph.
# 
# Additionally, the high frequency of the time series, with daily data points spanning a whole year (251 entries), can contribute to cluttering and overlapping data points. This makes it difficult to discern individual values or identify specific patterns within the data.
# 
# Moreover, the graph lacks sufficient context. The absence of axis labels, titles, or annotations makes it challenging to understand the meaning of the data or interpret the trends and patterns accurately.
# 
# To improve the interpretability of the graph, several steps can be taken. First, adjusting the scaling or presentation of the variables with smaller units or lower volatility can help make them more visible. Second, considering creating separate graphs for each data type, such as one graph for 'Open' and 'Close,' and several other graphs for the remaining data types. This can provide clearer visualizations and insights. Additionally, techniques like data aggregation or smoothing can be applied to reduce clutter and enhance readability. Finally, adding appropriate axis labels, titles, and annotations will provide the necessary context and reference points for understanding the overall trends and patterns in the time series.

# In[8]:


# F-b
df['Close'].plot()
plt.show()


# F-b-i.
# 
# The graph includes only one type of data: Close. The line for 'Close' was not visible in the first graph due to a large y-axis unit. However, in the current visualization, the y-axis unit has been auto-adjusted specifically for only one variable - Close, resulting in a clear depiction of the volatility and trend over the same time frame.
# 
# This graph is more easily interpretable than the previous one because it focuses solely on the 'Close' variable of the time series. By simplifying the representation and excluding other variables, it becomes easier to analyze the specific patterns and changes in the closing prices over time without distractions. Also, the adjusted y-axis allows the volatility of the variable was captured. This allows for a more precise interpretation of the time series data.

# In[9]:


# F-c
df['Volume'].plot()
plt.show()


# F-c-i.
# 
# A stock's trading volume refers to the number of shares traded between its daily open and close. This plot shows a volatile diagram for the daily trading volume, indicating fluctuations and changes over time. It appears that there is a certain seasonality in the trend. Specifically, we observed significantly higher trading volume in the months of April, July, October, and December compared to regular months. Higher trading volumes during certain months suggest increased investor activity and engagement in the stock during these months. 

# In[10]:


# F-d-i
# Plotting a subset for month April
start_date = '2023-04-01' 
end_date = '2023-04-30'  

subset_df = df.loc[start_date:end_date, 'Close']  

subset_df.plot()
plt.show()


# In[11]:


# F-d-ii
plt.figure(figsize=(10, 6))  
subset_df.plot(color='green', linestyle='--') 

plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Daily Close Prices for May 2023')

plt.show()


# In[12]:


# G-a-i
df_resampled = df['Close'].resample('M').mean() 

df_resampled.plot()

plt.xlabel('Date')
plt.ylabel('Mean Close Price')
plt.title('Monthly Mean Close Prices')
plt.show()


# G-a-ii
# 
# Weather forecasting models often require input data at specific time intervals. Resampling allows for adjusting the data to match the desired model time steps. For example, if a forecasting model operates at daily intervals but the available weather data is collected on an hourly basis, resampling can be used to aggregate the data to align with the model requirements. By resampling the data to a lower frequency, forecasters can obtain a clearer picture of the overall weather conditions for each day and identify more stable patterns. Additionally, weather data is collected from various sources and instruments, which may have different sampling frequencies. Resampling allows for aligning the data to a consistent time resolution for further analysis. 

# # Part II: A/B Testing Sales Promotion Strategies

# In[13]:


df_campaign = pd.read_csv('campaign_data.csv')
df_campaign


# In[14]:


# A 
# Generate a barplot to show the average SalesInThousands values, separated by the different promotion types
sns.barplot(x='Promotion', y='SalesInThousands', data=df_campaign, ci=None)

plt.xlabel('Promotion Type')
plt.ylabel('Average Sales (in Thousands)')
plt.title('Average Sales by Promotion Type')

plt.show()


# A-a.
# 
# The barplot visualizes the average SalesInThousands values generated by each promotion type. It shows that the average sales are similar for Promotion 1 (around 58,000), Promotion 2 (around 61,000), and Promotion 3 (around 55,000).

# In[15]:


# B
df_campaign['week'] = df_campaign['week'].astype('category')

sns.countplot(x='Promotion', hue='week', data=df_campaign)

plt.xlabel('Promotion Type')
plt.ylabel('Number of Instances')
plt.title('Number of Instances per Promotion Type')

plt.show()


# B-a.
# 
# As the gragh shows, the number of instances for each campaign is the same for every week, this suggests that the experiment design has an even distribution of promotions across time. This indicates a well-controlled experimental design where the 'week' variable is not confounded with the 'PromotionType' variable. In other words, the 'week' variable is not a confounding variable and it does not influence or interact with the promotion types. This design ensures that any observed differences in sales or outcomes can be more confidently attributed to the variation in the promotion types rather than the effect from 'week'.

# In[16]:


# C
summary_stats = df_campaign.groupby('Promotion')['AgeOfStore'].describe()

print(summary_stats)


# C-a.
# 
# There are 172 stores in Promotion 1, 188 stores in Promotion 2, and 188 stores in Promotion 3. The count of stores is similar across the three groups, suggesting that the sample sizes are relatively balanced.
# 
# Overall, based on these results, it appears that the age profile of the stores does not seem to be very different across the three groups. The average sales, variability (similar Standard Deviation), and range of sales are similar, reducing the risk of age profile becoming a confounding variable to the experiment. 

# In[17]:


# D 
import scipy.stats as stats

# Extract the sales values for each promotion 
sales_promo1 = df_campaign[df_campaign['Promotion'] == 1]['SalesInThousands']
sales_promo2 = df_campaign[df_campaign['Promotion'] == 2]['SalesInThousands']
sales_promo3 = df_campaign[df_campaign['Promotion'] == 3]['SalesInThousands']

# Perform pairwise t-tests
t_statistic_1_vs_2, p_value_1_vs_2 = stats.ttest_ind(sales_promo1, sales_promo2)
t_statistic_2_vs_3, p_value_2_vs_3 = stats.ttest_ind(sales_promo2, sales_promo3)
t_statistic_1_vs_3, p_value_1_vs_3 = stats.ttest_ind(sales_promo1, sales_promo3)

print("Promotion 1 vs. Promotion 2:")
print("T-Statistic:", t_statistic_1_vs_2)
print("P-Value:", p_value_1_vs_2)

print("\nPromotion 2 vs. Promotion 3:")
print("T-Statistic:", t_statistic_2_vs_3)
print("P-Value:", p_value_2_vs_3)

print("\nPromotion 1 vs. Promotion 3:")
print("T-Statistic:", t_statistic_1_vs_3)
print("P-Value:", p_value_1_vs_3)


# D-a.
# 
# The t-statistics and p-statistics for each pair is as below: 
# 
# Promotion 1 vs. Promotion 2:
# T-Statistic: -2.5230805833394934
# P-Value: 0.012065595012565785
# 
# Promotion 2 vs. Promotion 3:
# T-Statistic: 4.2191398016523785
# P-Value: 3.0807459102757044e-05
# 
# Promotion 1 vs. Promotion 3:
# T-Statistic: 1.5551383687293547
# P-Value: 0.12079667272313273

# D-b.
# 
# Promotion 1 vs. Promotion 2: The negative t-statistic of -2.523 suggests that Promotion 1 has, on average, lower sales than Promotion 2. In other words, Promotion 2 performs better than Promotion 1 in terms of sales. The p-value (0.0121) is less than the default significance level of 0.05. This indicates that there is strong evidence to conclude a statistically significant difference between the means of Promotion 1 and Promotion 2.
# 
# Promotion 2 vs. Promotion 3: The positive t-statistic of 4.219 suggests that the mean sales for Promotion 2 are higher than the mean sales for Promotion 3. This relatively large t-statistic indicates a big difference between the two promotions. The extremely small p-value (3.0807459102757044e-05) provides strong evidence to reject the null hypothesis and conclude that Promotion 2 significantly outperforms Promotion 3 in terms of sales.
# 
# Promotion 1 vs. Promotion 3: The t-statistic of 1.555 suggests a modest difference in mean sales between Promotion 1 and Promotion 3. However, the p-value of 0.12079667272313273 exceeds the significance level of 0.05, indicating that the observed difference is not statistically significant. Therefore, there is insufficient evidence to conclude that there is a meaningful difference between Promotion 1 and Promotion 3 in terms of sales.

# # Part III: Using a Statistical Test to Evaluate a Claim

# In[18]:


import numpy as np
from scipy.stats import chisquare

# A
# Recorded values
dice_values = np.array([1, 2, 3, 4, 5, 6])
observed_values = np.array([13, 7, 12, 8, 14, 6])

# Expected values 
expected_values = np.ones_like(dice_values) * (60 / 6)

# Perform chi-square goodness of fit test
chi2_stat, p_value = chisquare(observed_values, expected_values)

print("Chi-square statistic:", chi2_stat)
print("P-value:", p_value)


# A-a
# 
# The null hypothesis (H0) of the test in this case is that the observed dice rolls are fair thus follow the expected frequencies for a fair six-sided dice. 
# 
# The alternative hypothesis (Ha) is that the observed frequencies deviate significantly from the expected frequencies to be observed from a fair dice.

# A-b
# 
# Assuming that Lobster Land uses an alpha value of 0.05 for statistical tests, with a chi-square statistic of 5.800 and a p-value of 0.326, which is bigger than 0.05, we fail to reject the null hypothesis (H0: the observed dice rolls are fair thus follow the expected frequencies for a fair six-sided dice) in this case. Therefore, based on this test, we can not conclude that the observed frequencies of the dice rolls significantly deviate from the expected frequencies for a fair six-sided dice. Thus we cannot conclude whether the dice used are biased or unfair.

# In[19]:


# B
# Recorded values
observed_values_120 = np.array([26, 14, 24, 16, 28, 12])

expected_values_120 = np.ones_like(dice_values) * (120 / 6)

# Perform chi-square goodness of fit test
chi2_stat_120, p_value_120 = chisquare(observed_values_120, expected_values_120)

print("Chi-square statistic:", chi2_stat_120)
print("P-value:", p_value_120)


# B-a.
# 
# The null hypothesis (H0) of the test in this case is that the observed dice rolls are fair thus follow the expected frequencies for a fair six-sided dice. 
# 
# The alternative hypothesis (Ha) is that the observed frequencies deviate significantly from the expected frequencies to be observed from a fair dice.

# B-b.
# 
# Assuming that Lobster Land uses an alpha value of 0.05 for statistical tests, with a chi-square statistic of 11.600 and a p-value of 0.041, which is less than the chosen significance level (0.05), we have evidence to reject the null hypothesis (H0: that the observed dice rolls are fair thus follow the expected frequencies for a fair six-sided dice). This indicates that the observed frequencies deviate significantly from what would be expected from a fair six-sided die. Therefore, we can conclude that the dice is not fair.

# C-i.
# 
# Firstly, in both trials, the numbers 1, 3, and 5 appeared more frequently than 2, 4, and 6. Per the rule, if the roll results in 1, 3, or 5, the visitor will lose $12. 
# 
# Secondly, trial A involved 60 dice rolls and did not provide evidence of bias or unfairness in the dice. However, trial B, which included 120 dice rolls, suggests that the dice used in that trial is biased. This case of false negative in trial A, indicating that we should remain open to alternative hypotheses and be aware that the sample size can influence the test outcome and may fail to reject the null hypothesis.

# C-ii
# 
# To generate a chi-square statistic, we compare observed and expected values for each category.  We square the differences between the observed and expected values, and then divide that result by the expected value. The reason of the result difference could be sample size. A larger sample size provides more data points, thus can lead to more precise estimates of the expected frequencies and observed frequencies. With a smaller sample size, even small deviations from the expected frequencies can become statistically significant, making it less likely to miss detecting a true effect or bias. 
# 
# Additional, When the number of dice rolls is small, the observed frequencies may be more influenced by random variation. With smaller sample sizes, Random fluctuations can have a larger impact on the observed frequencies. Consequently, the test may not be able to distinguish between random variation and systematic deviations.

# D: 
# 
# Lobster Land should inform the traveling salesman that the dice and the game are unfair, and as a result, he will not be allowed to set up the dice game inside the park to cheat the park's guests. 
# 
# The decision is based on the findings from Trial B and the chi-square result, which revealed a significant deviation from the expected frequencies for a fair six-sided dice. This suggests that the dice used in the game is not fair. Lobster Land, being committed to maintaining its reputation and the trust of its customers, should not take the risk of setting up an unfair game inside the park for more financial benefits. Otherwise, it will lose its reputation and customers. 

# E. 
# 
# Normally you sleep for 7 hours, but on the second day after Children's Day you sleep for 9 hours because you enjoyed yourself at Disney. However, if we only focus on one week's worth of sleep data, the average will be significantly influenced by the 9-hour duration. On the contrary, when we record and calculate monthly or annual data, the average will more closely resemble the normal sleeping pattern. 
# 
# The same principle applies to the chi-square test. Having a larger sample size in a chi-square test provides us with greater confidence in the results because it helps to diminish the impact of chance occurrences or random variations, thereby offering a clearer depiction of whether there exists a genuine difference or bias in the data.

# # Part IV: Using Tableau to Build a Dashboard

# Tableau Dashboard:
# https://public.tableau.com/views/Hu654Assign4/Dashboard1?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link
# 
# Description:
# 
# What I did was experiment with different variables and plots to see if exploratory visualizations could provide insights beyond what could be obtained from examining the dataset directly. The specific steps involved first reconciling the data to the correct types, such as changing the date from a string to a date format. And then moving different parameters or measurements around and pair different elements together to observe the outcome. Also it's important to always check if the measured data is set in the right measurement (e.g. is total amount or average amount makes more sense). After exploring various plots and combinations of variables, I found the four plots displayed on the dashboard to be particularly interesting.
# 
# The plot titled "Correlation between Total Pax vs Total Revenue" presents the correlation between the number of guests and the total revenue generated. It appears that, for this dataset and scope, there is a positive correlation between the number of guests and the revenue generated. In other words, as the number of guests increases, the revenue also tends to increase. The plot labeled "Monthly Average Rev Trend" displays the average revenue for the months of May to September. It is evident that June has relatively higher daily revenue, contributing to a higher average monthly revenue. However, more complete data for May and September would be necessary to obtain a more accurate estimate of the monthly average. The bar plot titled "Average Revenue per Weekday" illustrates the differences in revenue for different weekdays. It clearly indicates that revenue is higher on weekends, specifically Friday, Saturday, and Sunday. Lastly, the plot "Average Number of Complaints per Weekday" demonstrates that, in general, Monday, Tuesday and Thursday tend to have a higher number of complaints. Overall, these visualizations provide valuable insights into the dataset and enable a better understanding of the relationships and trends within the sales data.
