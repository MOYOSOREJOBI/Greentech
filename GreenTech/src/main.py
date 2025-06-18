import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import json
import time
from datetime import datetime
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer

# Set the style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Configuration
OUTPUT_DIR = "./output"
MODELS_DIR = "./output/models"
PLOTS_DIR = "./output/plots"
RESULTS_DIR = "./output/results"
CSV_FILE = "data/energy_efficiency.csv"
# Create directories
for directory in [OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)


def run_entire_workflow(tune_hyperparams=True):
    """
    Run the entire machine learning workflow for the GreenTech Innovations project
    
    Parameters:
    -----------
    tune_hyperparams : bool
        Whether to perform hyperparameter tuning
        
    Returns:
    --------
    dict
        Dictionary containing all results and model details
    """
    print("\n===== GREENTECH INNOVATIONS ENERGY EFFICIENCY PREDICTION =====")
    print("Starting the machine learning workflow...")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: Hyperparameter tuning = {tune_hyperparams}")
    
    # Record start time
    start_time = time.time()
    
    # Step 1: Preprocess data
    print("\n----- DATA PREPROCESSING -----")
    preprocessor = DataPreprocessor(file_path=CSV_FILE)
    data = preprocessor.load_data()
    eda_results = preprocessor.explore_data()
    
    # Generate visualizations
    preprocessor.visualize_data(output_path=PLOTS_DIR)
    
    # Preprocess and split data
    (X_train, X_test, y_train_heating, y_test_heating,
     y_train_cooling, y_test_cooling) = preprocessor.preprocess_data()
    
    # Save processed data info
    data_info = {
        'data_shape': data.shape,
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'features': list(X_train.columns),
        'preprocessing_time': time.time() - start_time
    }
    
    with open(f"{RESULTS_DIR}/data_info.json", 'w') as f:
        json.dump(data_info, f, indent=4)
    
    # Step 2: Train models
    print("\n----- MODEL TRAINING AND EVALUATION -----")
    trainer = ModelTrainer(output_dir=MODELS_DIR)
    
    # Train models for heating load
    print("\nTraining models for Heating Load prediction:")
    heating_training_start = time.time()
    heating_models = trainer.train_all_models(
        X_train, X_test, 
        y_train_heating, y_test_heating, 
        'Heating_Load', 
        tune_hyperparams=tune_hyperparams
    )
    heating_training_time = time.time() - heating_training_start
    
    # Train models for cooling load
    print("\nTraining models for Cooling Load prediction:")
    cooling_training_start = time.time()
    cooling_models = trainer.train_all_models(
        X_train, X_test, 
        y_train_cooling, y_test_cooling, 
        'Cooling_Load', 
        tune_hyperparams=tune_hyperparams
    )
    cooling_training_time = time.time() - cooling_training_start
    
    # Step 3: Compare models and visualize feature importance
    print("\n----- MODEL COMPARISON AND ANALYSIS -----")
    
    # Compare models
    heating_comparison = trainer.compare_models(
        'Heating_Load', 
        output_file=f"{RESULTS_DIR}/heating_model_comparison.png"
    )
    cooling_comparison = trainer.compare_models(
        'Cooling_Load', 
        output_file=f"{RESULTS_DIR}/cooling_model_comparison.png"
    )
    
    # Visualize feature importance
    trainer.visualize_feature_importance(['Heating_Load', 'Cooling_Load'])
    
    # Step 4: Generate a summary report
    print("\n----- GENERATING SUMMARY REPORT -----")
    
    # Save model performances
    heating_performances = {
        k.replace('_heating_load', ''): {
            metric: float(v[metric]) if metric in ['mae', 'mse', 'rmse', 'r2'] else None
            for metric in ['mae', 'mse', 'rmse', 'r2']
        }
        for k, v in trainer.model_performances.items() if 'heating_load' in k
    }
    
    cooling_performances = {
        k.replace('_cooling_load', ''): {
            metric: float(v[metric]) if metric in ['mae', 'mse', 'rmse', 'r2'] else None
            for metric in ['mae', 'mse', 'rmse', 'r2']
        }
        for k, v in trainer.model_performances.items() if 'cooling_load' in k
    }
    
    # Identify best models
    best_heating_model = heating_comparison.loc[heating_comparison['R²'].idxmax()]['Model']
    best_cooling_model = cooling_comparison.loc[cooling_comparison['R²'].idxmax()]['Model']
    
    # Create final summary
    summary = {
        'project_info': {
            'name': 'GreenTech Innovations Energy Efficiency Prediction',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_runtime': time.time() - start_time
        },
        'data_info': data_info,
        'model_training': {
            'heating_training_time': heating_training_time,
            'cooling_training_time': cooling_training_time,
            'hyperparameter_tuning': tune_hyperparams
        },
        'best_models': {
            'heating_load': best_heating_model,
            'cooling_load': best_cooling_model
        },
        'model_performances': {
            'heating_load': heating_performances,
            'cooling_load': cooling_performances
        }
    }
    
    # Save summary
    with open(f"{RESULTS_DIR}/summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Print the final summary
    print("\n===== PROJECT SUMMARY =====")
    print(f"Total runtime: {(time.time() - start_time) / 60:.2f} minutes")
    print(f"Best model for Heating Load prediction: {best_heating_model}")
    print(f"Best model for Cooling Load prediction: {best_cooling_model}")
    print(f"All results saved to {RESULTS_DIR}")
    
    return summary


def predict_for_new_data(new_data, target='both'):
    """
    Make predictions for new data using the best trained models
    
    Parameters:
    -----------
    new_data : pd.DataFrame
        New data to make predictions on
    target : str
        'heating', 'cooling', or 'both' to specify which predictions to make
        
    Returns:
    --------
    dict
        Dictionary with predictions
    """
    # Load the summary to find best models
    with open(f"{RESULTS_DIR}/summary.json", 'r') as f:
        summary = json.load(f)
    
    # Determine which models to use
    best_heating_model = summary['best_models']['heating_load']
    best_cooling_model = summary['best_models']['cooling_load']
    
    # Load preprocessor for feature engineering
    preprocessor = DataPreprocessor(file_path=CSV_FILE)
    preprocessor.load_data()
    
    # Apply same preprocessing to new data
    if all(col in new_data.columns for col in ['Heating_Load', 'Cooling_Load']):
        X_new = new_data.drop(['Heating_Load', 'Cooling_Load'], axis=1)
    else:
        X_new = new_data.copy()
    
    # Apply feature engineering
    X_new = preprocessor._engineer_features(X_new)
    
    # Apply scaling using the saved scaler
    preprocessor.preprocess_data()  # This fits the scaler
    X_new_scaled = preprocessor._scale_features(X_new, fit=False)
    
    results = {'input_data': new_data.to_dict(orient='records')}
    
    # Make predictions
    if target in ['heating', 'both']:
        model_path = f"{MODELS_DIR}/{best_heating_model}_forest_heating_load.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            heating_pred = model.predict(X_new_scaled)
            results['heating_load_predictions'] = heating_pred.tolist()
        else:
            print(f"Model not found: {model_path}")
    
    if target in ['cooling', 'both']:
        model_path = f"{MODELS_DIR}/{best_cooling_model}_forest_cooling_load.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            cooling_pred = model.predict(X_new_scaled)
            results['cooling_load_predictions'] = cooling_pred.tolist()
        else:
            print(f"Model not found: {model_path}")
    
    return results


def demo_prediction():
    """
    Demonstrate prediction functionality with sample data
    """
    print("\n===== DEMONSTRATION: PREDICTION FOR NEW DATA =====")
    
    # Load the original dataset to get a few samples
    data = pd.read_csv(CSV_FILE)
    
    # Select 5 random samples and remove the target values
    sample_indices = np.random.choice(len(data), 5, replace=False)
    sample_data = data.iloc[sample_indices].copy()
    
    # Store the actual values for comparison
    actual_heating = sample_data['Heating_Load'].copy()
    actual_cooling = sample_data['Cooling_Load'].copy()
    
    # Make predictions
    predictions = predict_for_new_data(sample_data)
    
    # Compare predicted vs actual
    comparison = pd.DataFrame({
        'Actual_Heating': actual_heating.values,
        'Predicted_Heating': predictions.get('heating_load_predictions', []),
        'Heating_Diff': abs(actual_heating.values - np.array(predictions.get('heating_load_predictions', []))),
        'Actual_Cooling': actual_cooling.values,
        'Predicted_Cooling': predictions.get('cooling_load_predictions', []),
        'Cooling_Diff': abs(actual_cooling.values - np.array(predictions.get('cooling_load_predictions', [])))
    })
    
    print("\nPrediction Results (Sample):")
    print(comparison)
    
    # Calculate and print average error
    avg_heating_error = comparison['Heating_Diff'].mean()
    avg_cooling_error = comparison['Cooling_Diff'].mean()
    
    print(f"\nAverage prediction error for Heating Load: {avg_heating_error:.4f}")
    print(f"Average prediction error for Cooling Load: {avg_cooling_error:.4f}")
    
    # Save the demonstration results
    comparison.to_csv(f"{RESULTS_DIR}/prediction_demo.csv", index=False)
    print(f"Demonstration results saved to {RESULTS_DIR}/prediction_demo.csv")


if __name__ == "__main__":
    # Run the complete workflow
    summary = run_entire_workflow(tune_hyperparams=True)
    
    # Demonstrate prediction capability
    demo_prediction()
    
    print("\n===== PROJECT EXECUTION COMPLETE =====")
