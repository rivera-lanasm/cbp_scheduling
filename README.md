# CBP_SCHEDULING

## Overview

The CBP_SCHEDULING project aims to analyze and visualize airport customs data to optimize scheduling and reduce wait times. The project includes data processing, visualization, and modeling components to provide insights into passenger flow and service efficiency at airport terminals.

## Folder Structure


CBP_SCHEDULING/
├── __pycache__/
├── airport_scheduling_3.ipynb
├── airport_scheduling_sensitivity_analysis.ipynb
├── BI_solver_det.py
├── BI_solver_sto.py
├── data/
│   ├── jfk/
│   │   ├── jfk_2022.csv
│   │   ├── jfk_2023.csv
│   ├── mdw/
│   │   ├── mdw_06-2024.csv
│   │   ├── mdw_07-2024.csv
│   │   ├── mdw_08-2024.csv
│   │   ├── mdw_09-2024.csv
│   │   ├── mdw_10-2024.csv
├── data_viz.ipynb
├── hospital_env.py
├── model_arrival.ipynb
├── optimal_ctg_airport_env_hourly.csv
├── optimal_ctg_airport_env_time_of_day_April.csv
├── optimal_ctg_airport_env_time_of_day_August.csv
├── optimal_ctg_airport_env_time_of_day_December.csv
├── optimal_ctg_airport_env_time_of_day_February.csv
├── optimal_ctg_airport_env_time_of_day_January.csv
├── README.md



## Features

### Data Processing
- **Data Cleaning**: Scripts to clean and preprocess raw data from CSV files.
- **Data Aggregation**: Grouping and summarizing data by various dimensions such as month, week day, and hour.

### Visualization
- **Heatmaps**: Visualize average wait times and passenger counts using heatmaps.
- **Scatter Plots**: Analyze relationships between passenger counts and wait times.
- **Bar Charts**: Display distributions of arrivals and service rates.

### Modeling
- **Hospital Environment Simulation**: Simulate the hospital environment to visualize policies and cost-to-go functions.
- **Sensitivity Analysis**: Analyze the sensitivity of scheduling models to various parameters.

## Notebooks

- **airport_scheduling_3.ipynb**: Main notebook for scheduling analysis.
- **airport_scheduling_sensitivity_analysis.ipynb**: Notebook for sensitivity analysis.
- **data_viz.ipynb**: Notebook for data visualization.
- **model_arrival.ipynb**: Notebook for modeling passenger arrivals.

## Scripts

- **BI_solver_det.py**: Script for deterministic solver.
- **BI_solver_sto.py**: Script for stochastic solver.
- **hospital_env.py**: Script for simulating the hospital environment.

## Data

- **data/jfk/**: Contains JFK airport data for 2022 and 2023.
- **data/mdw/**: Contains Midway airport data for various months in 2024.

## Usage

1. **Data Preparation**: Ensure the data files are placed in the appropriate directories under `data`.
2. **Run Notebooks**: Open and run the Jupyter notebooks to perform analysis and visualization.
3. **Simulate Environment**: Use the `hospital_env.py` script to simulate the hospital environment and visualize policies.

## Requirements

- Python 3.x
- Jupyter Notebook
- pandas
- seaborn
- matplotlib
- numpy

## License

This project is licensed under the MIT License.

## Contact

For any questions or issues, please contact the project maintainers.
