import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # type: ignore
from sklearn.model_selection import GridSearchCV # type: ignore
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model # type: ignore
from keras.layers import Dense, Dropout, Input # type: ignore
from keras.callbacks import EarlyStopping # type: ignore
import os
import joblib
import time

class ModelTrainer:
    """
    Class to handle model training and evaluation for energy efficiency prediction
    """
    
    def __init__(self, output_dir="./models"):
        """
        Initialize the ModelTrainer
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the trained models
        """
        self.output_dir = output_dir
        self.models = {}
        self.model_performances = {}
        os.makedirs(output_dir, exist_ok=True)
        
    def train_decision_tree(self, X_train, y_train, target_name, tune_hyperparams=True):
        """
        Train a Decision Tree Regressor model
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target values
        target_name : str
            Name of the target (e.g., 'Heating_Load' or 'Cooling_Load')
        tune_hyperparams : bool
            Whether to perform hyperparameter tuning
            
        Returns:
        --------
        DecisionTreeRegressor
            Trained model
        """
        print(f"\n=== Training Decision Tree for {target_name} ===")
        
        if tune_hyperparams:
            print("Performing hyperparameter tuning...")
            param_grid = {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            dt = DecisionTreeRegressor(random_state=42)
            grid_search = GridSearchCV(
                dt, param_grid, cv=5, 
                scoring='neg_mean_squared_error', 
                verbose=1, n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            model = grid_search.best_estimator_
        else:
            model = DecisionTreeRegressor(random_state=42)
            model.fit(X_train, y_train)
        
        # Save the model
        model_name = f"decision_tree_{target_name.lower()}"
        self.models[model_name] = model
        joblib.dump(model, f"{self.output_dir}/{model_name}.pkl")
        
        print(f"Decision Tree model for {target_name} trained and saved.")
        return model
    
    def train_random_forest(self, X_train, y_train, target_name, tune_hyperparams=True):
        """
        Train a Random Forest Regressor model
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target values
        target_name : str
            Name of the target (e.g., 'Heating_Load' or 'Cooling_Load')
        tune_hyperparams : bool
            Whether to perform hyperparameter tuning
            
        Returns:
        --------
        RandomForestRegressor
            Trained model
        """
        print(f"\n=== Training Random Forest for {target_name} ===")
        
        if tune_hyperparams:
            print("Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, 
                scoring='neg_mean_squared_error', 
                verbose=1, n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            model = grid_search.best_estimator_
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
        
        # Save the model
        model_name = f"random_forest_{target_name.lower()}"
        self.models[model_name] = model
        joblib.dump(model, f"{self.output_dir}/{model_name}.pkl")
        
        print(f"Random Forest model for {target_name} trained and saved.")
        return model
    
    def train_neural_network(self, X_train, y_train, target_name):
        """
        Train an Artificial Neural Network model
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target values
        target_name : str
            Name of the target (e.g., 'Heating_Load' or 'Cooling_Load')
            
        Returns:
        --------
        keras.Model
            Trained model
        """
        print(f"\n=== Training Neural Network for {target_name} ===")
        
        # Convert to numpy arrays if they are not
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
            
        # Define the model architecture using the functional API
        input_dim = X_train.shape[1]
        inputs = Input(shape=(input_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='mean_squared_error'
        )
        
        # Set up early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=200,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Save the model
        model_name = f"neural_network_{target_name.lower()}"
        self.models[model_name] = model
        model.save(f"{self.output_dir}/{model_name}.keras")  # Updated to save with .keras extension
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'ANN Training History - {target_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{self.output_dir}/{model_name}_history.png")
        plt.close()
        
        print(f"Neural Network model for {target_name} trained and saved.")
        return model
    
    def evaluate_model(self, model, X_test, y_test, target_name, model_type):
        """
        Evaluate a trained model on test data
        
        Parameters:
        -----------
        model : trained model object
            The trained model to evaluate
        X_test : pd.DataFrame or np.array
            Test features
        y_test : pd.Series or np.array
            Test target values
        target_name : str
            Name of the target (e.g., 'Heating_Load' or 'Cooling_Load')
        model_type : str
            Type of the model (e.g., 'decision_tree', 'random_forest', 'neural_network')
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        print(f"\n=== Evaluating {model_type} for {target_name} ===")
        
        # Make predictions
        if model_type == 'neural_network':
            if isinstance(X_test, pd.DataFrame):
                X_test_array = X_test.values
            else:
                X_test_array = X_test
            y_pred = model.predict(X_test_array).flatten()
        else:
            y_pred = model.predict(X_test)
            
        # Convert to numpy arrays for consistency
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
            
        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Store the performance metrics
        performance = {
            'model_type': model_type,
            'target_name': target_name,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        # Save feature importance for tree-based models
        if model_type in ['decision_tree', 'random_forest']:
            performance['feature_importance'] = model.feature_importances_
        
        key = f"{model_type}_{target_name.lower()}"
        self.model_performances[key] = performance
        
        # Create prediction vs actual plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.title(f'{model_type.replace("_", " ").title()} - {target_name} Predictions')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{key}_predictions.png")
        plt.close()
        
        return performance
    
    def compare_models(self, target_name, output_file=None):
        """
        Compare the performance of different models for a specific target
        
        Parameters:
        -----------
        target_name : str
            Name of the target (e.g., 'Heating_Load' or 'Cooling_Load')
        output_file : str, optional
            Path to save the comparison plot
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with performance metrics for each model
        """
        print(f"\n=== Comparing Models for {target_name} ===")
        
        # Filter performances for the specified target
        target_performances = {
            k: v for k, v in self.model_performances.items() 
            if v['target_name'] == target_name
        }
        
        if not target_performances:
            print(f"No models found for target: {target_name}")
            return None
        
        # Create a comparison DataFrame
        comparison_data = []
        for model_name, perf in target_performances.items():
            comparison_data.append({
                'Model': model_name.split('_')[0],
                'MAE': perf['mae'],
                'MSE': perf['mse'],
                'RMSE': perf['rmse'],
                'R²': perf['r2']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df)
        
        # Create comparison plots
        metrics = ['MAE', 'RMSE', 'R²']
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            sns.barplot(x='Model', y=metric, data=comparison_df, ax=axes[i])
            axes[i].set_title(f'{metric} Comparison')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.savefig(f"{self.output_dir}/model_comparison_{target_name.lower()}.png")
        
        plt.close()
        
        return comparison_df
    
    def train_all_models(self, X_train, X_test, y_train, y_test, target_name, 
                         tune_hyperparams=True):
        """
        Train all models (Decision Tree, Random Forest, Neural Network) for a target
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Test features
        y_train : pd.Series
            Training target values
        y_test : pd.Series
            Test target values
        target_name : str
            Name of the target (e.g., 'Heating_Load' or 'Cooling_Load')
        tune_hyperparams : bool
            Whether to perform hyperparameter tuning for tree-based models
            
        Returns:
        --------
        dict
            Dictionary with all trained models
        """
        # Train Decision Tree
        start_time = time.time()
        dt_model = self.train_decision_tree(X_train, y_train, target_name, tune_hyperparams)
        dt_time = time.time() - start_time
        self.evaluate_model(dt_model, X_test, y_test, target_name, 'decision_tree')
        
        # Train Random Forest
        start_time = time.time()
        rf_model = self.train_random_forest(X_train, y_train, target_name, tune_hyperparams)
        rf_time = time.time() - start_time
        self.evaluate_model(rf_model, X_test, y_test, target_name, 'random_forest')
        
        # Train Neural Network
        start_time = time.time()
        nn_model = self.train_neural_network(X_train, y_train, target_name)
        nn_time = time.time() - start_time
        self.evaluate_model(nn_model, X_test, y_test, target_name, 'neural_network')
        
        # Compare models
        self.compare_models(target_name)
        
        # Print training times
        print("\n=== Training Times ===")
        print(f"Decision Tree: {dt_time:.2f} seconds")
        print(f"Random Forest: {rf_time:.2f} seconds")
        print(f"Neural Network: {nn_time:.2f} seconds")
        
        return {
            'decision_tree': dt_model,
            'random_forest': rf_model,
            'neural_network': nn_model
        }
    
    def visualize_feature_importance(self, target_names):
        """
        Visualize feature importance for tree-based models
        
        Parameters:
        -----------
        target_names : list
            List of target names to visualize feature importance for
        """
        for target_name in target_names:
            for model_type in ['random_forest']:  # Best to use Random Forest for feature importance
                key = f"{model_type}_{target_name.lower()}"
                
                if key in self.model_performances and 'feature_importance' in self.model_performances[key]:
                    # Get feature importance and feature names
                    importance = self.model_performances[key]['feature_importance']
                    
                    # Get feature names from the first model we find
                    model_name = f"{model_type}_{target_name.lower()}"
                    if model_name in self.models:
                        if hasattr(self.models[model_name], 'feature_names_in_'):
                            feature_names = self.models[model_name].feature_names_in_
                        else:
                            # If feature names are not available, use indices
                            feature_names = [f'Feature {i}' for i in range(len(importance))]
                        
                        # Create a DataFrame for better sorting and visualization
                        feature_importance = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importance
                        }).sort_values('Importance', ascending=False)
                        
                        # Plot feature importance
                        plt.figure(figsize=(12, 8))
                        sns.barplot(x='Importance', y='Feature', data=feature_importance)
                        plt.title(f'Feature Importance - {model_type.title()} for {target_name}')
                        plt.tight_layout()
                        plt.savefig(f"{self.output_dir}/{key}_feature_importance.png")
                        plt.close()
                        
                        print(f"Feature importance visualization saved for {model_type} - {target_name}")
                    else:
                        print(f"Model {model_name} not found in trained models dictionary")


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataPreprocessor
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    (X_train, X_test, y_train_heating, y_test_heating,
     y_train_cooling, y_test_cooling) = preprocessor.preprocess_data()
    
    # Train models for heating load
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, X_test, y_train_heating, y_test_heating, 'Heating_Load')
    
    # Train models for cooling load
    trainer.train_all_models(X_train, X_test, y_train_cooling, y_test_cooling, 'Cooling_Load')
    
    # Visualize feature importance
    trainer.visualize_feature_importance(['Heating_Load', 'Cooling_Load'])
