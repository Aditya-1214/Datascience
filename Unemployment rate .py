#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r'C:\Users\DELL\Downloads\Unemployment_rate.csv') 

print(data.head())
print(data.isnull().sum())

print(data.describe())

print(data.columns)

data.columns = data.columns.str.strip()

data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='Date', y='Estimated Unemployment Rate (%)', hue='Region')
plt.title('Unemployment Rate over Time by Region')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Area', y='Estimated Unemployment Rate (%)')
plt.title('Unemployment Rate Distribution by Area')
plt.xlabel('Area')
plt.ylabel('Unemployment Rate (%)')
plt.tight_layout()
plt.show()

andhra_pradesh_data = data[data['Region'] == 'Andhra Pradesh']

plt.figure(figsize=(10, 6))
sns.lineplot(data=andhra_pradesh_data, x='Date', y='Estimated Unemployment Rate (%)')
plt.title('Unemployment Rate in Andhra Pradesh over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

may_2020_data = data[data['Date'] == '2020-05-31']
print(may_2020_data[['Region', 'Estimated Unemployment Rate (%)']])

data.to_csv('cleaned_unemployment_data.csv', index=False)


# In[ ]:




