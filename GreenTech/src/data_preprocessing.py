"""
Data Preprocessing Module for GreenTech Innovations Energy Efficiency Project
Author: Moyosoreoluwa Ogunjobi
Date: April 12, 2025

This module handles all data preprocessing tasks including:
- Data loading and inspection
- Cleaning and normalization
- Feature engineering
- Train-test splitting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    """
    Class to handle all data preprocessing operations
    """
    
    def __init__(self, file_path="energy_efficiency.csv"):
        """
        Initialize the DataPreprocessor
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing the energy efficiency data
        """
        self.file_path = file_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train_heating = None
        self.y_test_heating = None
        self.y_train_cooling = None
        self.y_test_cooling = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """
        Load the dataset from CSV file
        
        Returns:
        --------
        pd.DataFrame
            The loaded dataframe
        """
        print(f"Loading data from {self.file_path}...")
        self.data = pd.read_csv(self.file_path)
        print(f"Dataset loaded with shape: {self.data.shape}")
        return self.data
    
    def explore_data(self):
        """
        Perform exploratory data analysis and return basic statistics
        
        Returns:
        --------
        dict
            Dictionary containing various EDA results
        """
        if self.data is None:
            self.load_data()
            
        # Basic information
        print("\n=== Dataset Information ===")
        print(f"Number of samples: {self.data.shape[0]}")
        print(f"Number of features: {self.data.shape[1]}")
        
        # Check for missing values
        missing_values = self.data.isnull().sum()
        print("\n=== Missing Values ===")
        print(missing_values)
        
        # Summary statistics
        print("\n=== Summary Statistics ===")
        desc_stats = self.data.describe()
        print(desc_stats)
        
        # Return a dictionary with all the EDA results
        eda_results = {
            'missing_values': missing_values,
            'desc_stats': desc_stats,
            'data_sample': self.data.head(),
            'data_shape': self.data.shape
        }
        
        return eda_results
    
    def visualize_data(self, output_path="./plots"):
        """
        Generate and save visualization plots for data insights
        
        Parameters:
        -----------
        output_path : str
            Directory path to save the plots
        """
        if self.data is None:
            self.load_data()
            
        import os
        os.makedirs(output_path, exist_ok=True)
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation = self.data.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap of Features')
        plt.tight_layout()
        plt.savefig(f"{output_path}/correlation_heatmap.png")
        plt.close()
        
        # Distribution of target variables
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.histplot(self.data['Heating_Load'], kde=True, ax=axes[0])
        axes[0].set_title('Distribution of Heating Load')
        sns.histplot(self.data['Cooling_Load'], kde=True, ax=axes[1])
        axes[1].set_title('Distribution of Cooling Load')
        plt.tight_layout()
        plt.savefig(f"{output_path}/target_distributions.png")
        plt.close()
        
        # Pairplot for selected features
        selected_features = ['Relative_Compactness', 'Surface_Area', 'Overall_Height', 
                           'Glazing_Area', 'Heating_Load', 'Cooling_Load']
        sns.pairplot(self.data[selected_features])
        plt.suptitle('Pairplot of Selected Features', y=1.02)
        plt.savefig(f"{output_path}/features_pairplot.png")
        plt.close()
        
        print(f"Visualizations saved to {output_path}")
        
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Preprocess the data including:
        - Handling any missing values
        - Feature scaling
        - Train-test split
        
        Parameters:
        -----------
        test_size : float
            Proportion of the dataset to be used as test set
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        tuple
            Preprocessed training and test sets
        """
        if self.data is None:
            self.load_data()
            
        print("\n=== Preprocessing Data ===")
        
        # Check for and handle missing values
        if self.data.isnull().sum().any():
            print("Handling missing values...")
            # For numerical features, fill with median
            for col in self.data.select_dtypes(include=['float64', 'int64']).columns:
                self.data[col].fillna(self.data[col].median(), inplace=True)
        
        # Split features and targets
        X = self.data.drop(['Heating_Load', 'Cooling_Load'], axis=1)
        y_heating = self.data['Heating_Load']
        y_cooling = self.data['Cooling_Load']
        
        # Feature engineering
        self._engineer_features(X)
        
        # Split into train and test sets
        X_train, X_test, y_train_heating, y_test_heating = train_test_split(
            X, y_heating, test_size=test_size, random_state=random_state
        )
        _, _, y_train_cooling, y_test_cooling = train_test_split(
            X, y_cooling, test_size=test_size, random_state=random_state
        )
        
        # Scale the features
        X_train = self._scale_features(X_train)
        X_test = self._scale_features(X_test, fit=False)
        
        # Store the processed data
        self.X_train = X_train
        self.X_test = X_test
        self.y_train_heating = y_train_heating
        self.y_test_heating = y_test_heating
        self.y_train_cooling = y_train_cooling
        self.y_test_cooling = y_test_cooling
        
        print("Data preprocessing completed.")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return (X_train, X_test, y_train_heating, y_test_heating, 
                y_train_cooling, y_test_cooling)
    
    def _scale_features(self, X, fit=True):
        """
        Scale features using StandardScaler
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features to scale
        fit : bool
            Whether to fit the scaler on this data
            
        Returns:
        --------
        pd.DataFrame
            Scaled features
        """
        if fit:
            return pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns
            )
        else:
            return pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns
            )
    
    def _engineer_features(self, X):
        """
        Generate new features based on domain knowledge
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features dataframe
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with engineered features
        """
        # Calculate volume as a new feature
        if 'Surface_Area' in X.columns and 'Overall_Height' in X.columns:
            X['Volume'] = X['Surface_Area'] * X['Overall_Height'] / 3
        
        # Calculate wall-to-roof ratio
        if 'Wall_Area' in X.columns and 'Roof_Area' in X.columns:
            X['Wall_Roof_Ratio'] = X['Wall_Area'] / X['Roof_Area']
        
        # Calculate relative height (height normalized by surface area)
        if 'Overall_Height' in X.columns and 'Surface_Area' in X.columns:
            X['Relative_Height'] = X['Overall_Height'] / X['Surface_Area']
        
        # Feature interaction between compactness and glazing
        if 'Relative_Compactness' in X.columns and 'Glazing_Area' in X.columns:
            X['Compactness_Glazing'] = X['Relative_Compactness'] * X['Glazing_Area']
        
        return X
    
    def get_processed_data(self):
        """
        Get the preprocessed data
        
        Returns:
        --------
        tuple
            Preprocessed training and test sets
        """
        if self.X_train is None:
            self.preprocess_data()
            
        return (self.X_train, self.X_test, self.y_train_heating, self.y_test_heating,
                self.y_train_cooling, self.y_test_cooling)


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data()
    eda_results = preprocessor.explore_data()
    preprocessor.visualize_data()
    
    # Preprocess data and get train/test sets
    (X_train, X_test, y_train_heating, y_test_heating,
     y_train_cooling, y_test_cooling) = preprocessor.preprocess_data()
