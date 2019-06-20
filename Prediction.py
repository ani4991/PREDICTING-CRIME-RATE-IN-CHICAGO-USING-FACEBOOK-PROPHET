#1: IMPORTING DATA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from fbprophet import Prophet

# dataframes creation for both training and testing datasets
chicago_df_1 = pd.read_csv('Chicago_Crimes_2005_to_2007.csv', error_bad_lines=False)
chicago_df_2 = pd.read_csv('Chicago_Crimes_2008_to_2011.csv', error_bad_lines=False)
chicago_df_3 = pd.read_csv('Chicago_Crimes_2012_to_2017.csv', error_bad_lines=False)

chicago_df = pd.concat([chicago_df_1, chicago_df_2, chicago_df_3], ignore_index=False, axis=0)

#2: EXPLORING THE DATASET

# the head of the training dataset
chicago_df.head()

# the last elements in the training dataset
chicago_df.tail(20)

# check null elements are contained in the data
plt.figure(figsize=(10,10))
sns.heatmap(chicago_df.isnull(), cbar = False, cmap = 'YlGnBu')

# ID Case Number Date Block IUCR Primary Type Description Location Description Arrest Domestic Beat District Ward Community Area FBI Code X Coordinate Y Coordinate Year Updated On Latitude Longitude Location
chicago_df.drop(['Unnamed: 0', 'Case Number', 'Case Number', 'IUCR', 'X Coordinate', 'Y Coordinate','Updated On','Year', 'FBI Code', 'Beat','Ward','Community Area', 'Location', 'District', 'Latitude' , 'Longitude'], inplace=True, axis=1)

print(chicago_df)

# Assembling a datetime by rearranging the dataframe column "Date".
chicago_df.Date = pd.to_datetime(chicago_df.Date, format='%m/%d/%Y %I:%M:%S %p')

# setting the index to be the date
chicago_df.index = pd.DatetimeIndex(chicago_df.Date)

print(chicago_df['Primary Type'].value_counts())

plt.figure(figsize = (15, 10))
sns.countplot(y= 'Primary Type', data = chicago_df, order = chicago_df['Primary Type'].value_counts().iloc[:15].index)
plt.show()

plt.figure(figsize = (15, 10))
sns.countplot(y= 'Location Description', data = chicago_df, order = chicago_df['Location Description'].value_counts().iloc[:15].index)
plt.show()

print(chicago_df.resample('Y').size())

# Resample method for frequency conversion and resampling of time series.

plt.plot(chicago_df.resample('Y').size())
plt.title('Crimes Count Per Year')
plt.xlabel('Years')
plt.ylabel('Number of Crimes')
plt.show()

# Resample method for frequency conversion and resampling of time series.

plt.plot(chicago_df.resample('M').size())
plt.title('Crimes Count Per Month')
plt.xlabel('Months')
plt.ylabel('Number of Crimes')
plt.show()

# Resample method for frequency conversion and resampling of time series.

plt.plot(chicago_df.resample('Q').size())
plt.title('Crimes Count Per Quarter')
plt.xlabel('Quarters')
plt.ylabel('Number of Crimes')
plt.show()

#3: PREPARING THE DATA
chicago_prophet = chicago_df.resample('M').size().reset_index()

print(chicago_prophet)

chicago_prophet.columns = ['Date', 'Crime Count']

chicago_prophet_df = pd.DataFrame(chicago_prophet)
print(chicago_prophet_df)

#4: MAKE PREDICTIONS

print(chicago_prophet_df.columns)

chicago_prophet_df_final = chicago_prophet_df.rename(columns={'Date':'ds', 'Crime Count':'y'})
print(chicago_prophet_df_final)

m = Prophet()
m.fit(chicago_prophet_df_final)

# Forcasting into the future
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
print(forecast)

figure = m.plot(forecast, xlabel='Date', ylabel='Crime Rate')
figure3 = m.plot_components(forecast)








