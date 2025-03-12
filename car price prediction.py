#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

data = pd.read_csv(r"C:\Users\DELL\Downloads\car data.csv")
print(data.columns)
df = pd.get_dummies(data, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)


X = df.drop(['Selling_Price', 'Car_Name'], axis=1) 
y = df['Selling_Price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_score = cross_val_score(best_rf, X_scaled, y, cv=5, scoring='neg_mean_squared_error').mean()

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
print(f"Cross-Validation Score: {cv_score}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c=(y_pred - y_test), cmap='coolwarm', alpha=0.7, edgecolors='k')
plt.colorbar(label='Difference between Actual and Predicted Prices') 

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--', label='Perfect Prediction')

plt.xlabel('Actual Prices', fontsize=12, color='darkblue')
plt.ylabel('Predicted Prices', fontsize=12, color='darkblue')
plt.title('Actual vs Predicted Car Prices', fontsize=14, color='darkblue')
plt.legend(loc='upper left', fontsize=10)

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[ ]:




