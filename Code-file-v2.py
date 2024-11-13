#Import all the files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

#Read the files
cab_df = pd.read_csv("cab_rides.csv")
weather = pd.read_csv("weather.csv")

#Convert to datetime
cab_df['date_time'] = pd.to_datetime(cab_df['time_stamp']/1000, unit='s')
weather['date_time'] = pd.to_datetime(weather['time_stamp'], unit='s')

# Create a new column for merging, and imputing rain columns with 0
cab_df['merge_date'] = cab_df['source'].astype(str) + " - " + cab_df['date_time'].dt.date.astype(str) + " - " + cab_df['date_time'].dt.hour.astype(str)
weather['merge_date'] = weather['location'].astype(str) + " - " + weather['date_time'].dt.date.astype(str) + " - " + weather['date_time'].dt.hour.astype(str)

# Group by 'merge_date' and calculate the mean for numeric columns only
numeric_weather = weather.select_dtypes(include='number')
groupby_value = weather[['merge_date']].join(numeric_weather).groupby(['merge_date']).mean().reset_index()
# Fill NaN values in the 'rain' column with 0
groupby_value['rain'].fillna(0, inplace=True)
# Display the grouped dataframe
groupby_value.head()

# Merge two dataframe and drop NAN
groupby_value.index = groupby_value['merge_date']
merged_df = cab_df.join(groupby_value,on=['merge_date'],rsuffix ='_w')
merged_df.dropna(inplace=True)
merged_df.head(5)
#################################################################################################################################
################################################################Plotting Graphs##################################################

# plotting distance against price
fig , ax = plt.subplots(figsize = (12,5))
ax.plot(merged_df[merged_df['cab_type'] == 'Lyft'].groupby('distance').price.mean().index,
        merged_df[merged_df['cab_type'] == 'Lyft'].groupby('distance').price.mean(),
        label = 'Lyft', color='deeppink')

ax.plot(merged_df[merged_df['cab_type'] == 'Uber'].groupby('distance').price.mean().index,
        merged_df[merged_df['cab_type'] =='Uber'].groupby('distance').price.mean(),
        label = 'Uber', color='blue')

ax.set_title('The Average Price by distance', fontsize= 15, fontweight='bold')
ax.set(xlabel = 'Distance', ylabel = 'Price' )
ax.legend()
plt.show()

# plotting Vehicle Type against average price
uber_order =[ 'UberPool','WAV', 'UberX', 'UberXL', 'Black','Black SUV' ]
lyft_order = ['Shared', 'Lyft', 'Lyft XL', 'Lux', 'Lux Black', 'Lux Black XL']
fig, ax = plt.subplots(1,2, figsize = (12,4))
ax1 = sns.barplot(x = merged_df[merged_df['cab_type'] == 'Uber'].name,
                  y = merged_df[merged_df['cab_type'] == 'Uber'].price ,
                  ax = ax[0], order = uber_order,palette='Blues')

ax2 = sns.barplot(x = merged_df[merged_df['cab_type'] == 'Lyft'].name,
                  y = merged_df[merged_df['cab_type'] == 'Lyft'].price ,
                  ax = ax[1], order = lyft_order,palette='Reds')

ax1.set(xlabel = 'Vehicle Type', ylabel = 'Average Price')
ax2.set(xlabel = 'Vehicle Type', ylabel = 'Average Price')

ax1.set_title('The Uber Average Prices by Vehicle Type', fontweight='bold')
ax2.set_title('The Lyft Average Prices by Vehicle Type', fontweight='bold')

plt.show()

# plotting source against price
sns.catplot(x='source', y='price', data=merged_df, kind='box', hue='cab_type', sym='', height=4, aspect=2, palette='Blues', dodge=True)
plt.tick_params(axis='x', rotation=45)
plt.suptitle("Distribution of Pickup Location vs Price", y=1.05, fontweight='bold')
plt.show()
###################################################################################################################################
############################################################ Feature Selection ####################################################
# Selecting features
merged_df = merged_df[['distance','cab_type','destination','source','surge_multiplier','price','name','date_time','merge_date','temp','clouds','pressure','rain','humidity','wind']]

# Creating time period variable and mapping
merged_df['hour'] = merged_df['date_time'].dt.hour.astype(str)
mapping = {
    '6': 'morning','7' : 'morning','8' : 'morning','9' : 'morning',
    '10' : 'noon', '11' : 'noon','12' : 'noon', '13' : 'noon',
    '14' : 'afternoon', '15' : 'afternoon', '16' : 'afternoon', '17' : 'afternoon',
    '18' : 'evening', '19' : 'evening', '20' : 'evening', '21' : 'evening',
    '22' : 'night', '23' : 'night', '0' : 'night', '1' : 'night',
    '2' : 'night', '3' : 'late_night', '4' : 'late_night', '5' : 'late_night' }
merged_df['time_period'] = merged_df['hour'].replace(mapping)

merged_df.drop(columns=['date_time','merge_date','hour'],axis=1, inplace=True)
merged_df.head(5)


# Subsetting dataframe into uber and lyft
df_lyft = merged_df[merged_df['cab_type']=='Lyft'].copy()
df_uber = merged_df[merged_df['cab_type']=='Uber'].copy()

# Creating target and features
X_lyft = df_lyft.drop('price',axis=1)
y_lyft = df_lyft['price'].copy()

X_uber = df_uber.drop('price',axis=1)
y_yber = df_uber['price'].copy()

# Splitting data into training and testing for Uber and Lyft dataframes
from sklearn.model_selection import train_test_split
X_train_lyft, X_test_lyft, y_train_lyft, y_test_lyft = train_test_split(X_lyft, y_lyft, test_size=0.2, random_state=42)
X_train_uber, X_test_uber, y_train_uber, y_test_uber = train_test_split(X_uber, y_yber, test_size=0.2, random_state=42)

################################################Pre-preocessing#################################################################
# Creating preprocessing pipeline
from sklearn import set_config
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#For categorical features
set_config(display='diagram')

cat_attribs = ["cab_type", "destination", "source", "name", "time_period"]
num_attribs = ["distance", "temp", "clouds", "pressure", "rain", "humidity", "wind"]

preprocess_pipeline = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first"), cat_attribs),
        ("num", StandardScaler(), num_attribs),])

preprocess_pipeline

# Test if it's preprocessing
print(X_train_lyft.shape)
X_train_lyft_prepared = preprocess_pipeline.fit_transform(X_train_lyft)
print(X_train_lyft_prepared.shape)
################################################################################################################################
#########################################################Data Modelling#########################################################

# Linear Regression Bayes Search for Uber dataframe
from sklearn.linear_model import LinearRegression
from skopt import BayesSearchCV
from skopt.space import Categorical

sparse_columns = ['cab_type', 'destination', 'source', 'name', 'time_period']

# One-hot encoding categorical features without creating sparse matrices
X_train_uber_selected = X_train_uber[sparse_columns].to_numpy()
uber_bayes_encoder = OneHotEncoder(sparse=False)
X_train_uber_encoded = pd.get_dummies(X_train_uber[sparse_columns])
# Concatenating encoded features with the original dataset
X_train_uber_bayes_final = pd.concat([X_train_uber.drop(columns=sparse_columns), X_train_uber_encoded], axis=1)

param_space = {
    'fit_intercept': Categorical([True, False]),
    'copy_X': Categorical([True, False]),
    'positive': Categorical([True, False])
}

lin_uber_bayesian_search = BayesSearchCV(
    estimator=LinearRegression(),
    search_spaces=param_space,
    n_iter=8,
    cv=3,
    scoring='neg_root_mean_squared_error',
    random_state=42
)

lin_uber_bayesian_search.fit(X_train_uber_bayes_final, y_train_uber)

# Presenting result
lin_uber_bayes_res = pd.DataFrame(lin_uber_bayesian_search.cv_results_)
lin_uber_bayes_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
lin_uber_bayes_res.filter(regex = '(^param_|mean_test_score)', axis=1)


# Linear Regression Bayes Search for Lyft dataframe

X_train_lyft_selected = X_train_lyft[sparse_columns].to_numpy()
lyft_bayes_encoder = OneHotEncoder(sparse=False)
X_train_lyft_encoded = pd.get_dummies(X_train_lyft[sparse_columns])
X_train_lyft_bayes_final = pd.concat([X_train_lyft.drop(columns=sparse_columns), X_train_lyft_encoded], axis=1)


param_space = {
    'fit_intercept': Categorical([True, False]),
    'copy_X': Categorical([True, False]),
    'positive': Categorical([True, False])
}

lin_lyft_bayesian_search = BayesSearchCV(
    estimator=LinearRegression(),
    search_spaces=param_space,
    n_iter=8,
    cv=3,
    scoring='neg_root_mean_squared_error',
    random_state=42
)

lin_lyft_bayesian_search.fit(X_train_lyft_bayes_final, y_train_lyft)

# Presenting result
lin_lyft_bayes_res = pd.DataFrame(lin_lyft_bayesian_search.cv_results_)
lin_lyft_bayes_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
lin_lyft_bayes_res.filter(regex = '(^param_|mean_test_score)', axis=1)




# XGB Bayes Search for Uber dataframe

import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

xgb_uber = make_pipeline(preprocess_pipeline, XGBRegressor())

param_bayes = {
    'xgbregressor__n_estimators': (50, 200),
    'xgbregressor__max_depth': (3, 7),
    'xgbregressor__colsample_bytree': (0.8, 1.0)
}

xgb_uber_bayes_search = BayesSearchCV(xgb_uber, search_spaces=param_bayes, n_iter=10, cv=2, scoring='neg_root_mean_squared_error', n_jobs=-1)
xgb_uber_bayes_search.fit(X_train_uber, y_train_uber)

# Evaluate Performance - Bayes Search
y_pred_bayes = xgb_uber_bayes_search.predict(X_test_uber)
rmse_bayes = np.sqrt(mean_squared_error(y_test_uber, y_pred_bayes))
print(f"Bayes Search RMSE: {rmse_bayes}")




# XGB Bayes Search for lyft dataframe
xgb_lyft = make_pipeline(preprocess_pipeline, XGBRegressor())
param_bayes = {
    'xgbregressor__n_estimators': (50, 200),
    'xgbregressor__max_depth': (3, 7),
    'xgbregressor__subsample': (0.8, 1.0),
}

xgb_lyft_bayes_search = BayesSearchCV(xgb_lyft, search_spaces=param_bayes, n_iter=10, cv=2, scoring='neg_root_mean_squared_error', n_jobs=-1)
xgb_lyft_bayes_search.fit(X_train_lyft, y_train_lyft)

# Evaluate Performance - Bayes Search
y_pred_bayes = xgb_lyft_bayes_search.predict(X_test_lyft)
rmse_bayes = np.sqrt(mean_squared_error(y_test_lyft, y_pred_bayes))
print(f"Bayes Search RMSE: {rmse_bayes}")


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

###############################################################################################################################
##################################################Demand Prediction############################################################
# Prepare data for demand prediction
# Selecting relevant features for predicting demand
merged_df['date_hour'] = merged_df['date_time'].dt.floor('H')
demand_data = merged_df[['source', 'date_hour', 'temp', 'rain', 'clouds', 'humidity', 'distance', 'price', 'surge_multiplier']]
demand_df = merged_df.groupby(['source', 'date_hour']).size().reset_index(name='demand')

# Merge the calculated demand back into the main dataset
demand_data = merged_df.merge(demand_df, on=['source', 'date_hour'], how='left')


# Encode categorical variables
demand_data = pd.get_dummies(demand_data, columns=['source','cab_type'], drop_first=True)

merged_df
# Split data
X_demand = demand_data.drop(['demand', 'date_hour'], axis=1)
y_demand = demand_data['demand']
X_train_demand, X_test_demand, y_train_demand, y_test_demand = train_test_split(X_demand, y_demand, test_size=0.2, random_state=42)

# Train Random Forest model for demand prediction
demand_model = RandomForestRegressor(random_state=0)
demand_model.fit(X_train_demand, y_train_demand)

# Predict and evaluate
y_pred_demand = demand_model.predict(X_test_demand)
mae_demand = mean_absolute_error(y_test_demand, y_pred_demand)
print(f"Demand Prediction MAE: {mae_demand:.2f}")

#######################################################################################################################################
###########################################################Hotspot Clustering##########################################################

from sklearn.cluster import KMeans

# Use demand and location features to find hotspots
hotspot_data = merged_df[['source', 'distance', 'surge_multiplier', 'temp', 'rain', 'humidity']]
hotspot_data = pd.get_dummies(hotspot_data, columns=['source'], drop_first=True)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=0)
merged_df['hotspot_cluster'] = kmeans.fit_predict(hotspot_data)

# Display cluster counts to identify high-demand zones
hotspot_counts = merged_df['hotspot_cluster'].value_counts()
print("Hotspot Cluster Counts:")
print(hotspot_counts)

###################################################################################################################################
##################################################### Route Optimization###########################################################
#Route Optimization
import networkx as nx
import pandas as pd
import numpy as np
# Create a DataFrame with location names and their approximate latitude and longitude in Boston
locations = {
    "location": [
        "Haymarket Square", "Back Bay", "North End", "North Station", "Beacon Hill",
        "Boston University", "Fenway", "South Station", "Theatre District", "West End",
        "Financial District", "Northeastern University"
    ],
    "latitude": [42.3637, 42.3493, 42.3648, 42.3663, 42.3583, 42.3505, 42.3467, 42.3519, 42.3513, 42.3656, 42.3559, 42.3398],
    "longitude": [-71.0585, -71.0843, -71.0546, -71.0621, -71.0709, -71.1087, -71.0972, -71.0550, -71.0645, -71.0637, -71.0568, -71.0892]
}

locations_df = pd.DataFrame(locations)



def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in miles
    R = 3958.8
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Calculate pairwise distances between locations
distances = []
for i, loc1 in locations_df.iterrows():
    for j, loc2 in locations_df.iterrows():
        if i < j:
            distance = haversine(loc1['latitude'], loc1['longitude'], loc2['latitude'], loc2['longitude'])
            distances.append((loc1['location'], loc2['location'], round(distance, 2)))

# Create a DataFrame for the calculated distances
distances_df = pd.DataFrame(distances, columns=['start', 'end', 'distance'])



# Initialize a graph and add edges with distances
G = nx.Graph()
for _, row in distances_df.iterrows():
    G.add_edge(row['start'], row['end'], weight=row['distance'])

# Define start and end points for route optimization (for example)
start_point = 'Haymarket Square'
end_point = 'Fenway'

# Calculate the shortest path and distance
shortest_path = nx.dijkstra_path(G, start_point, end_point, weight='weight')
shortest_distance = nx.dijkstra_path_length(G, start_point, end_point, weight='weight')

print("Optimal Route:", shortest_path)
print("Shortest Distance:", shortest_distance)

##################################################################################################################################
###################################################### Metric Calculation#########################################################

demand_df = merged_df.groupby(['source', 'hour']).size().reset_index(name='demand')
surge_frequency = merged_df[merged_df['surge_multiplier'] > 1.0].groupby(['source', 'hour']).size().reset_index(name='surge_frequency')
merged_df['fare_per_mile'] = merged_df['price'] / merged_df['distance']
average_fare_per_mile = merged_df.groupby(['source', 'hour'])['fare_per_mile'].mean().reset_index(name='average_fare_per_mile')
average_duration = merged_df.groupby(['source', 'hour'])['duration'].mean().reset_index(name='average_duration')
weather_impact = merged_df[merged_df['rain'] > 0].groupby(['source'])[['price', 'surge_multiplier']].mean().reset_index()
weather_impact.columns = ['source', 'average_price_rain', 'average_surge_rain']
peak_hours = merged_df.groupby('hour').size().reset_index(name='ride_count').sort_values(by='ride_count', ascending=False)
canceled_rides = merged_df[merged_df['status'] == 'canceled'].groupby(['source', 'hour']).size()
total_rides = merged_df.groupby(['source', 'hour']).size()
cancellation_rate = (canceled_rides / total_rides).reset_index(name='cancellation_rate').fillna(0)
average_waiting_time = merged_df.groupby(['source', 'hour'])['waiting_time'].mean().reset_index(name='average_waiting_time')
# Merge all metrics into a single DataFrame
metrics_df = demand_df.merge(surge_frequency, on=['source', 'hour'], how='left')
metrics_df = metrics_df.merge(average_fare_per_mile, on=['source', 'hour'], how='left')
metrics_df = metrics_df.merge(average_duration, on=['source', 'hour'], how='left')
metrics_df = metrics_df.merge(weather_impact, on='source', how='left')
metrics_df = metrics_df.merge(cancellation_rate, on=['source', 'hour'], how='left')
metrics_df = metrics_df.merge(average_waiting_time, on=['source', 'hour'], how='left')

# Display the final DataFrame
metrics_df.head()

#######################################################################################################################################