# GreenTech Innovations Energy Efficiency Prediction

A machine learning solution for predicting building energy efficiency (Heating and Cooling Load) using various building parameters.

## Project Overview

This project was developed for GreenTech Innovations, a renewable energy company seeking to optimize energy consumption and resource allocation. The solution uses machine learning models to predict heating and cooling loads based on building characteristics, allowing for more efficient energy management.

## Dataset

The project uses the "Energy Efficiency" dataset, which contains building parameters such as:
- Relative Compactness
- Surface Area
- Wall Area
- Roof Area
- Overall Height
- Orientation
- Glazing Area
- Glazing Area Distribution

And the target variables:
- Heating Load
- Cooling Load

## Project Structure

```
├── data_preprocessing.py       # Data preprocessing and feature engineering
├── model_training.py           # Model implementation and evaluation
├── main.py                     # Main execution script
├── energy_efficiency.csv       # Dataset (placed in the root directory)
├── models/                     # Directory for saved models
├── plots/                      # Directory for data visualizations
├── results/                    # Directory for analysis results
└── output/                     # Directory for additional output files
```

## Machine Learning Models

Three models are implemented and compared:

1. **Decision Tree Regression**: A non-linear model that partitions the data space into regions and makes predictions based on the region a data point falls into.

2. **Random Forest Regression**: An ensemble model that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

3. **Artificial Neural Network (ANN)**: A deep learning model that captures complex non-linear relationships between features.

## Requirements

The project requires the following Python packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- joblib

## How to Run

1. Ensure the `energy_efficiency.csv` file is in the root directory
2. Execute the main script:

```bash
python main.py
```

This will:
- Preprocess the data
- Train all three models for both heating and cooling load prediction
- Evaluate and compare model performance
- Generate visualizations and summary reports
- Demonstrate prediction capabilities with sample data

## Files and Functions

### `data_preprocessing.py`

Contains the `DataPreprocessor` class with methods for:
- Data loading and exploration
- Data cleaning and normalization
- Feature engineering
- Train-test splitting

### `model_training.py`

Contains the `ModelTrainer` class with methods for:
- Training Decision Tree models
- Training Random Forest models
- Training Neural Network models
- Model evaluation and comparison
- Feature importance visualization

### `main.py`

Implements the complete workflow:
- Initializes and executes data preprocessing
- Trains and evaluates all models
- Compares model performance
- Generates visualizations and reports
- Provides prediction functionality for new data

## Results and Output

After running the script, check the following directories:

- `models/`: Contains the saved trained models
- `plots/`: Contains data visualizations such as correlation heatmaps and feature distributions
- `results/`: Contains model comparison results and summary information

## Prediction for New Data

The system can make predictions for new buildings. See the `predict_for_new_data()` function in `main.py` for implementation details.

## Author

Moyosoreoluwa Ogunjobi
