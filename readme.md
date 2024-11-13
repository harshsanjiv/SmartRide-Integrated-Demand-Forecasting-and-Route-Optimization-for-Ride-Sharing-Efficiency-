# •	SmartRide Optimizer - Integrated fare prediction,demand forecasting, and route optimization for ride sharing efficiency

This project analyzes ride-sharing data for Boston, using machine learning models to predict demand and fare, identify high-demand hotspots, and optimize routes between key locations.

## Key Objectives

1. **Demand Prediction**: Predict hourly demand across different locations.
2. **Fare Prediction**: Estimate ride fares based on factors such as distance and surge pricing.
3. **Hotspot Identification**: Use clustering to identify high-demand zones for better resource allocation.
4. **Route Optimization**: Find optimal routes using real location data in Boston.

## Project Structure

- **data/**: Contains datasets like `cab_rides.csv` and `weather.csv`.
- **scripts/**: Python scripts for data processing, model training, and evaluation.
- **README.md**: Project overview and documentation.
- **requirements.txt**: List of dependencies for easy setup.
  
## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Boston_RideSharing_Project.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the scripts or notebooks:
    ```bash
    python scripts/demand_prediction.py
    ```

## Results and Insights

- **Demand Prediction**: The model achieved an MAE of ... for demand prediction.
- **Fare Prediction**: The fare prediction model showed an RMSE of ...
- **Hotspot Analysis**: Identified high-demand zones, primarily around areas like ...
- **Route Optimization**: The shortest paths were calculated between key locations using Dijkstra’s algorithm.

## License

This project is licensed under the MIT License.
