# Importing the libraries
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings('ignore')

# Importing the dataset
weather_df = pd.read_csv('weatherforecasting.csv', parse_dates=['date_time'], index_col='date_time')
print(weather_df.head(5))
print(weather_df.columns)
print(weather_df.shape)
print(weather_df.describe())
print(weather_df.isnull().any())

weather_df_num = weather_df.loc[:, ['maxtempC', 'mintempC', 'cloudcover', 'humidity', 'tempC', 'sunHour', 'HeatIndexC', 'precipMM', 'pressure', 'windspeedKmph']]
print(weather_df_num.head())
print(weather_df_num.shape)
print(weather_df_num.columns)

# Visualizing the data
weather_df_num.plot(subplots=True, figsize=(25, 20))
plt.show()

weather_df_num['2019':'2020'].resample('D').fillna(method='pad').plot(subplots=True, figsize=(25, 20))
plt.show()

weth = weather_df_num['2019':'2020']
print(weth.head())

# Splitting the dataset into the Training set and Test set
weather_y = weather_df_num.pop("tempC")
weather_x = weather_df_num
train_X, test_X, train_y, test_y = train_test_split(weather_x, weather_y, test_size=0.2, random_state=4)

# Visualizing relationship between features and target variable
plt.scatter(weth.mintempC, weth.tempC)
plt.xlabel("Minimum Temperature")
plt.ylabel("Temperature")
plt.show()

plt.scatter(weth.HeatIndexC, weth.tempC)
plt.xlabel("Heat Index")
plt.ylabel("Temperature")
plt.show()

plt.scatter(weth.pressure, weth.tempC)
plt.xlabel("Pressure")
plt.ylabel("Temperature")
plt.show()

# Training the Linear Regression model
lin_model = LinearRegression()
lin_model.fit(train_X, train_y)
lin_prediction = lin_model.predict(test_X)

print("Linear Regression - MAE: %.2f" % np.mean(np.absolute(lin_prediction - test_y)))
print("Linear Regression - MSE: %.2f" % np.mean((lin_prediction - test_y) ** 2))
print("Linear Regression - RMSE: %.2f" % np.sqrt(mean_squared_error(test_y, lin_prediction)))
print("Linear Regression - R2-score: %.2f" % r2_score(test_y, lin_prediction))
print()

# Training the Decision Tree Regression model
tree_model = DecisionTreeRegressor(random_state=0)
tree_model.fit(train_X, train_y)
tree_prediction = tree_model.predict(test_X)

print("Decision Tree - MAE: %.2f" % np.mean(np.absolute(tree_prediction - test_y)))
print("Decision Tree - MSE: %.2f" % np.mean((tree_prediction - test_y) ** 2))
print("Decision Tree - RMSE: %.2f" % np.sqrt(mean_squared_error(test_y, tree_prediction)))
print("Decision Tree - R2-score: %.2f" % r2_score(test_y, tree_prediction))
print()

# Training the Random Forest Regression model
forest_model = RandomForestRegressor(max_depth=90, random_state=0, n_estimators=100)
forest_model.fit(train_X, train_y)
forest_prediction = forest_model.predict(test_X)

print("Random Forest - MAE: %.2f" % np.mean(np.absolute(forest_prediction - test_y)))
print("Random Forest - MSE: %.2f" % np.mean((forest_prediction - test_y) ** 2))
print("Random Forest - RMSE: %.2f" % np.sqrt(mean_squared_error(test_y, forest_prediction)))
print("Random Forest - R2-score: %.2f" % r2_score(test_y, forest_prediction))
print()

# Plotting predictions for Linear Regression
plt.figure(figsize=(15, 6))
plt.scatter(test_y, lin_prediction, color='blue', label='Predictions')
plt.plot([min(test_y), max(test_y)], [min(test_y), max(test_y)], color='red', linestyle='--', label='Ideal line')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Linear Regression - True vs. Predicted')
plt.legend()
plt.grid(True)
plt.show()

# Plotting predictions for Random Forest Regression
plt.figure(figsize=(15, 6))
plt.scatter(test_y, forest_prediction, color='green', label='Predictions')
plt.plot([min(test_y), max(test_y)], [min(test_y), max(test_y)], color='red', linestyle='--', label='Ideal line')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Random Forest Regression - True vs. Predicted')
plt.legend()
plt.grid(True)
plt.show()

# Visualizing the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(tree_model, filled=True, feature_names=train_X.columns, max_depth=3, fontsize=10)
plt.title('Decision Tree (Displayed to depth of 3)')
plt.show()
