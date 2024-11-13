# Boston Ride-Sharing Data Analysis and Route Optimization

## Project Overview
This project explores ride-sharing data from the Boston area, focusing on enhancing demand forecasting, fare prediction, and operational efficiency. By leveraging machine learning and optimization techniques, we aim to improve resource allocation, pricing strategies, and route planning for ride-sharing services.

## Problem Statement and Objectives

The ride-sharing industry faces multiple challenges that affect operational efficiency, customer satisfaction, and economic performance. This project addresses five key objectives:

1. **Predict Demand**: Develop predictive models to forecast demand at different locations and times to optimize driver availability.
2. **Fare Prediction**: Build models to estimate ride fares based on distance, surge multiplier, and weather conditions, helping users and operators anticipate pricing.
3. **Identify High-Demand Hotspots**: Cluster key areas in Boston to recognize high-demand zones, facilitating better resource allocation.
4. **Route Optimization for Drivers**: Determine the most efficient routes between popular pickup and drop-off locations to reduce travel time and fuel costs.
5. **Analysis of External Factors on Ride Demand**: Examine the impact of weather and time variables on demand, fare fluctuations, and cancellation rates.

## Importance of the Study

- **Economic Efficiency**: Accurate demand predictions and optimized routes improve operational efficiency and reduce costs.
- **Environmental Impact**: Optimized routing reduces unnecessary mileage, leading to lower fuel consumption and emissions.
- **Customer Satisfaction**: Enhanced demand forecasting and pricing strategies improve customer experience by reducing wait times and price uncertainty.
- **Strategic Planning**: The insights support long-term planning for resource allocation and policy decisions in urban transportation.

## Dataset

The dataset provides a comprehensive view of ride-sharing operations in Boston, including ride details, weather conditions, and fare data. Key attributes include:

- **Ride Details**: `source`, `destination`, `cab_type`, `name`, `distance`, `price`, `surge_multiplier`, `status`
- **Timing Information**: `time_stamp`, `date_time`, `hour`, `date_hour`
- **Weather Conditions**: `temp`, `rain`, `clouds`, `humidity`, `wind`

## Key Analyses

### 1. Demand Prediction
- **Objective**: Predict hourly demand across different locations in Boston to optimize driver availability.
- **Methods**: Used Random Forest regression with time-based and weather features.
- **Outcome**: Achieved a Mean Absolute Error (MAE) of X%, providing accurate demand forecasts for high-traffic periods.

### 2. Fare Prediction
- **Objective**: Estimate ride fares based on factors such as distance, surge, and weather conditions.
- **Methods**: Employed Random Forest regression to predict fare based on ride attributes and weather.
- **Outcome**: Achieved an MAE of Y%, enabling better fare estimation and improved pricing strategies.

### 3. Identification of High-Demand Hotspots
- **Objective**: Identify clusters of high-demand locations for strategic driver allocation.
- **Methods**: Used K-means clustering on ride locations to identify areas with frequent requests.
- **Outcome**: Identified four key hotspots in Boston, helping to inform driver placement and service expansion areas.

### 4. Route Optimization for Drivers
- **Objective**: Determine the shortest path between popular pickup and drop-off locations to reduce travel time.
- **Methods**: Implemented Dijkstra’s Algorithm with NetworkX using realistic distances based on Boston locations.
- **Outcome**: Provided optimal routes between high-demand locations, reducing driver travel times and improving operational efficiency.

### 5. Analysis of External Factors on Ride Demand
- **Objective**: Analyze the impact of weather, time, and location on ride demand and pricing.
- **Methods**: Correlation analysis and regression models to assess relationships between weather conditions and ride demand.
- **Outcome**: Identified strong relationships between demand, weather conditions, and surge pricing, supporting dynamic allocation and pricing strategies.

## Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/harshsanjiv/SmartRide-Integrated-Demand-Forecasting-and-Route-Optimization-for-Ride-Sharing-Efficiency-.git
    ```

2. **Navigate to the project directory**:
    ```bash
    cd RideSharing_Project
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the scripts or notebooks**:
    ```bash
    python scripts/Code-file-v2.py
    ```

## Results and Insights

- **Demand Prediction**: The model trains for demand prediction, effectively identifying high-demand periods and locations.
- **Fare Prediction**: The fare prediction model  offers accurate fare estimates based on ride characteristics and weather.
- **Hotspot Analysis**: Identified key high-demand zones primarily around areas such as [example zones].
- **Route Optimization**: Determined the shortest paths between high-demand areas, reducing driver travel times and improving efficiency.


