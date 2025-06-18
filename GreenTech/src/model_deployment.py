"""
Model Deployment Module for GreenTech Innovations Energy Efficiency Project
Author: Moyosoreoluwa Ogunjobi
Date: April 12, 2025

This module simulates deployment of the trained models in a production environment:
- Loading trained models
- Creating a prediction API
- Batch prediction for multiple buildings
- Real-time monitoring 
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import time
from datetime import datetime
from flask import Flask, request, jsonify
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deployment.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("energy_efficiency_deployment")

class ModelDeployment:
    """
    Class to handle the deployment of trained models in production.
    """
    
    def __init__(self, models_dir="./models", results_dir="./results"):
        """
        Initialize the ModelDeployment class
        
        Parameters:
        -----------
        models_dir : str
            Directory containing the trained models
        results_dir : str
            Directory containing the results and summary
        """
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.models = {}
        self.preprocessing_params = {}
        self.feature_names = []
        self.summary = None
        
        # Load the summary information
        self._load_summary()
        
        # Load the best models
        self._load_models()
    
    def _load_summary(self):
        """
        Load the summary information from the project
        """
        try:
            summary_path = os.path.join(self.results_dir, "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    self.summary = json.load(f)
                logger.info("Summary loaded successfully.")
                
                # Get feature names from the summary
                if 'data_info' in self.summary and 'features' in self.summary['data_info']:
                    self.feature_names = self.summary['data_info']['features']
            else:
                logger.warning(f"Summary file not found: {summary_path}")
        except Exception as e:
            logger.error(f"Error loading summary: {e}")
            
    def _load_models(self):
        """
        Load the best models from the models directory
        """
        if not self.summary or 'best_models' not in self.summary:
            logger.error("Cannot load models: Summary not available or missing best_models information")
            return
        
        try:
            # Load heating load model
            best_heating_model = self.summary['best_models']['heating_load']
            heating_model_path = os.path.join(self.models_dir, f"{best_heating_model}_heating_load.pkl")
            if os.path.exists(heating_model_path):
                self.models['heating_load'] = joblib.load(heating_model_path)
                logger.info(f"Heating load model loaded: {best_heating_model}")
            else:
                logger.warning(f"Heating load model not found: {heating_model_path}")
            
            # Load cooling load model
            best_cooling_model = self.summary['best_models']['cooling_load']
            cooling_model_path = os.path.join(self.models_dir, f"{best_cooling_model}_cooling_load.pkl")
            if os.path.exists(cooling_model_path):
                self.models['cooling_load'] = joblib.load(cooling_model_path)
                logger.info(f"Cooling load model loaded: {best_cooling_model}")
            else:
                logger.warning(f"Cooling load model not found: {cooling_model_path}")
                
            # Load scaler and preprocessing parameters (in a real system, these would be saved separately)
            # For simplicity, we're assuming the preprocessing parameters are part of the model
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def preprocess_input(self, input_data):
        """
        Preprocess input data to match the format expected by the models
        
        Parameters:
        -----------
        input_data : pd.DataFrame or dict
            Input data to preprocess
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed data ready for prediction
        """
        try:
            # Convert dict to DataFrame if necessary
            if isinstance(input_data, dict):
                input_data = pd.DataFrame([input_data])
            elif isinstance(input_data, list):
                input_data = pd.DataFrame(input_data)
                
            # Ensure all required features are present
            missing_features = [f for f in self.feature_names if f not in input_data.columns]
            if missing_features:
                logger.warning(f"Missing features in input data: {missing_features}")
                raise ValueError(f"Input data missing required features: {missing_features}")
                
            # Apply feature engineering (simplified version - in production this would use saved transformers)
            processed_data = input_data.copy()
            
            # Calculate derived features as done in the data preprocessing module
            if 'Surface_Area' in processed_data.columns and 'Overall_Height' in processed_data.columns:
                processed_data['Volume'] = processed_data['Surface_Area'] * processed_data['Overall_Height'] / 3
            
            if 'Wall_Area' in processed_data.columns and 'Roof_Area' in processed_data.columns:
                processed_data['Wall_Roof_Ratio'] = processed_data['Wall_Area'] / processed_data['Roof_Area']
            
            if 'Overall_Height' in processed_data.columns and 'Surface_Area' in processed_data.columns:
                processed_data['Relative_Height'] = processed_data['Overall_Height'] / processed_data['Surface_Area']
            
            if 'Relative_Compactness' in processed_data.columns and 'Glazing_Area' in processed_data.columns:
                processed_data['Compactness_Glazing'] = processed_data['Relative_Compactness'] * processed_data['Glazing_Area']
                
            # In a real system, we would apply scaling using a saved scaler
            # For this simulation, we'll assume the models can handle unscaled data
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing input data: {e}")
            raise
    
    def predict(self, input_data):
        """
        Make predictions for input data
        
        Parameters:
        -----------
        input_data : pd.DataFrame or dict
            Input data for predictions
            
        Returns:
        --------
        dict
            Dictionary containing predictions
        """
        try:
            # Preprocess the input data
            processed_data = self.preprocess_input(input_data)
            
            results = {'timestamp': datetime.now().isoformat()}
            
            # Make predictions for heating load
            if 'heating_load' in self.models:
                heating_predictions = self.models['heating_load'].predict(processed_data)
                results['heating_load_predictions'] = heating_predictions.tolist()
            else:
                logger.warning("Heating load model not available for prediction")
                
            # Make predictions for cooling load
            if 'cooling_load' in self.models:
                cooling_predictions = self.models['cooling_load'].predict(processed_data)
                results['cooling_load_predictions'] = cooling_predictions.tolist()
            else:
                logger.warning("Cooling load model not available for prediction")
                
            # Add input data for reference
            results['input_data'] = processed_data.to_dict(orient='records')
            
            # Log the prediction
            logger.info(f"Predictions made for {len(processed_data)} samples")
            
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def batch_predict(self, input_file, output_file=None):
        """
        Make predictions for a batch of data from a CSV file
        
        Parameters:
        -----------
        input_file : str
            Path to the input CSV file
        output_file : str, optional
            Path to save the predictions
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the input data and predictions
        """
        try:
            logger.info(f"Starting batch prediction for file: {input_file}")
            
            # Load the input data
            input_data = pd.read_csv(input_file)
            logger.info(f"Loaded {len(input_data)} samples from {input_file}")
            
            # Make predictions
            predictions = self.predict(input_data)
            
            # Combine input data with predictions
            if 'heating_load_predictions' in predictions:
                input_data['Predicted_Heating_Load'] = predictions['heating_load_predictions']
            if 'cooling_load_predictions' in predictions:
                input_data['Predicted_Cooling_Load'] = predictions['cooling_load_predictions']
                
            # Save predictions if output file is specified
            if output_file:
                input_data.to_csv(output_file, index=False)
                logger.info(f"Predictions saved to {output_file}")
                
            return input_data
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise
    
    def run_prediction_service(self, host='0.0.0.0', port=5000):
        """
        Run a Flask API service for real-time predictions
        
        Parameters:
        -----------
        host : str
            Host to run the service on
        port : int
            Port to run the service on
        """
        app = Flask("GreenTech_Energy_Efficiency_API")
        
        @app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({'status': 'healthy', 'models_loaded': list(self.models.keys())})
        
        @app.route('/predict', methods=['POST'])
        def predict_endpoint():
            try:
                # Get JSON data from request
                data = request.json
                
                # Make predictions
                predictions = self.predict(data)
                
                return jsonify(predictions)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        @app.route('/batch_predict', methods=['POST'])
        def batch_predict_endpoint():
            try:
                # Check if file was uploaded
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400
                    
                file = request.files['file']
                
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                # Save the uploaded file
                temp_input = 'temp_input.csv'
                temp_output = 'temp_output.csv'
                file.save(temp_input)
                
                # Make batch predictions
                result_df = self.batch_predict(temp_input, temp_output)
                
                # Return the path to the output file
                return jsonify({
                    'message': f'Batch prediction completed for {len(result_df)} samples',
                    'output_file': temp_output
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        # Start the Flask app
        logger.info(f"Starting prediction service on {host}:{port}")
        app.run(host=host, port=port)


# Example usage in simulated production environment
def simulate_deployment():
    """
    Simulate deployment of the models in production
    """
    print("\n===== SIMULATING MODEL DEPLOYMENT =====")
    
    # Initialize the deployment
    deployment = ModelDeployment()
    
    # Simulate real-time predictions
    print("\n--- Real-time Prediction Simulation ---")
    sample_building = {
        'Relative_Compactness': 0.75,
        'Surface_Area': 650.0,
        'Wall_Area': 300.0,
        'Roof_Area': 200.0,
        'Overall_Height': 3.5,
        'Orientation': 2,
        'Glazing_Area': 0.2,
        'Glazing_Area_Distribution': 3
    }
    
    try:
        prediction = deployment.predict(sample_building)
        print("\nPrediction for a single building:")
        print(f"Heating Load: {prediction.get('heating_load_predictions', ['N/A'])[0]:.2f} kWh")
        print(f"Cooling Load: {prediction.get('cooling_load_predictions', ['N/A'])[0]:.2f} kWh")
    except Exception as e:
        print(f"Error making prediction: {e}")
    
    # Simulate batch predictions
    print("\n--- Batch Prediction Simulation ---")
    
    # Create a sample batch of buildings
    sample_batch = pd.DataFrame([
        {
            'Relative_Compactness': 0.70,
            'Surface_Area': 630.0,
            'Wall_Area': 290.0,
            'Roof_Area': 190.0,
            'Overall_Height': 3.5,
            'Orientation': 4,
            'Glazing_Area': 0.1,
            'Glazing_Area_Distribution': 1
        },
        {
            'Relative_Compactness': 0.82,
            'Surface_Area': 680.0,
            'Wall_Area': 310.0,
            'Roof_Area': 210.0,
            'Overall_Height': 7.0,
            'Orientation': 3,
            'Glazing_Area': 0.3,
            'Glazing_Area_Distribution': 2
        },
        {
            'Relative_Compactness': 0.95,
            'Surface_Area': 700.0,
            'Wall_Area': 330.0,
            'Roof_Area': 170.0,
            'Overall_Height': 7.0,
            'Orientation': 5,
            'Glazing_Area': 0.4,
            'Glazing_Area_Distribution': 4
        }
    ])
    
    # Save the sample batch to a CSV file
    sample_batch_file = "sample_batch.csv"
    sample_batch.to_csv(sample_batch_file, index=False)
    
    try:
        result_df = deployment.batch_predict(sample_batch_file, "batch_prediction_results.csv")
        print("\nBatch prediction results:")
        print(result_df[['Relative_Compactness', 'Overall_Height', 'Predicted_Heating_Load', 'Predicted_Cooling_Load']])
    except Exception as e:
        print(f"Error in batch prediction: {e}")
    
    print("\n===== DEPLOYMENT SIMULATION COMPLETE =====")
    print("In a real production environment, you would:")
    print("1. Set up a containerized API using Docker")
    print("2. Deploy to cloud services (AWS, GCP, Azure)")
    print("3. Set up monitoring and alerting")
    print("4. Implement authentication and API rate limiting")
    print("5. Create a CI/CD pipeline for model updates")


if __name__ == "__main__":
    simulate_deployment()
    
    # To run the API service, uncomment the following lines:
    # deployment = ModelDeployment()
    # deployment.run_prediction_service(port=5000)
