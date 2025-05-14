#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Data wrangling
import pandas as pd

# Scientific
import numpy as np
from scipy import stats
import math

# Model loading
import pickle
import json
import os
import sys
import argparse
from datetime import datetime

# XGBoost
from xgboost import XGBRegressor

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

# Default paths for model files - using relative paths based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "comprehensive_model.pkl")
MODEL_CONFIG_PATH = os.path.join(SCRIPT_DIR, "comprehensive_model.txt")

# D10 transformation and inverse transformation class
class D10Transformer:
    """
    Implement a transformation similar to bestNormalize for D10 values
    """
    def __init__(self):
        # Original D10 data used to fit the transformation model
        self.original_d10 = np.array([1.7,9.8,4,0.9,0.59,0.55,11,9,4.9,8.5,8.5,4.2,11,7,11,4,9,2.5,4,9.5,8,4.8,11,
                                    7.5,5.1,12.5,7.4,11,9.5,6.5,3,10.6,6.5,11.5,6,3,5,8,12,9,8,11,7.4,9,10,11,11.5,
                                    11,14,11.5,5.3,7.5,5,3.5,2.8,10,2.5,3.5,7.5,5.3,5.1,4,13,13.5,10,11,1.7,1.7,0.2,
                                    0.2,0.4,0.64,1,8.5,9.1,9.5,9.4,6,6,9.5,5,2,12,12,7.6,4.2,7.3,5.2,5.5,2.9,2.9,
                                    0.47,0.69,4,0.64,0.63,0.81,3,2,2.1,6,1,2.4,1.5,2.8,4,3,0.4,0.63,5,1.6,12,15,
                                    11.5,7,10.3,0.5,1,10,4,2.2,12,3,9,5.5,9.8,0.29,10,8])
        
        self._fit_transform_model()
        
    def _fit_transform_model(self):
        """Fit transformation model using percentile mapping"""
        sorted_d10 = np.sort(self.original_d10)
        n = len(sorted_d10)
        
        # Store sorted values and corresponding percentiles
        self.sorted_values = sorted_d10
        self.percentiles = np.linspace(0, 1, n)
    
    def inverse_transform(self, transformed_values):
        """
        Transform values back to original scale using percentile mapping
        """
        result = np.zeros_like(transformed_values, dtype=float)
        
        # Convert from normal distribution to percentiles
        norm_percentiles = stats.norm.cdf(transformed_values)
        
        # Map percentiles back to original data range
        for i, p in enumerate(norm_percentiles):
            # Find closest percentile point
            if p <= self.percentiles[0]:
                result[i] = self.sorted_values[0]
            elif p >= self.percentiles[-1]:
                result[i] = self.sorted_values[-1]
            else:
                # Find interval containing p
                idx_left = np.where(self.percentiles <= p)[0][-1]
                idx_right = np.where(self.percentiles >= p)[0][0]
                
                # If exact match
                if idx_left == idx_right:
                    result[i] = self.sorted_values[idx_left]
                else:
                    # Linear interpolation
                    left_p = self.percentiles[idx_left]
                    right_p = self.percentiles[idx_right]
                    left_val = self.sorted_values[idx_left]
                    right_val = self.sorted_values[idx_right]
                    
                    # Calculate interpolated value
                    weight = (p - left_p) / (right_p - left_p)
                    result[i] = left_val + weight * (right_val - left_val)
        
        return result

def load_model(filepath):
    """Load model from specified path"""
    try:
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
        print(f"Model successfully loaded from: {filepath}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_model_config(filepath):
    """Load model configuration from JSON file"""
    try:
        with open(filepath, 'r') as file:
            config = json.load(file)
        print(f"Model configuration successfully loaded from: {filepath}")
        return config
    except Exception as e:
        print(f"Error loading model configuration: {e}")
        return None

def predict_d10_values(model, validation_data, transformer, feature_order=None):
    """
    Predict D10 values for new bacteria samples and transform back to original scale
    
    Parameters:
    -----------
    model : XGBRegressor
        Trained XGBoost model
    validation_data : pandas.DataFrame
        Validation data containing features only
    transformer : D10Transformer
        Transformer object for inverting predictions to original scale
    feature_order : list, optional
        Ordered list of features used during training
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with bacterial samples and their predicted D10 values (both transformed and original)
    """
    # Make a copy of validation data to avoid modifying the original
    data = validation_data.copy()
    
    # Extract sample names
    sample_names = data['Sample'].values
    data = data.drop('Sample', axis=1)
    
    # Ensure features are in the same order as during training if specified
    if feature_order is not None:
        # Verify all required features are present
        missing_features = set(feature_order) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing features in validation data: {missing_features}")
        
        # Reorder columns to match training data
        data = data[feature_order]
    
    print("Making predictions in transformed space...")
    # Make predictions (these are in transformed space)
    transformed_predictions = model.predict(data)
    
    print("Performing inverse transformation to original scale...")
    # Inverse transform predictions to original scale
    original_predictions = transformer.inverse_transform(transformed_predictions)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Sample': sample_names,
        'Transformed_D10': transformed_predictions,
        'Predicted_D10': original_predictions
    })
    
    return results

def visualize_predictions(predictions, save_path=None):
    """
    Create a bar plot of predicted D10 values with reference line at y=1
    and color-coded bars (red for D10 >= 1, blue for D10 < 1)
    
    Parameters:
    -----------
    predictions : pandas.DataFrame
        DataFrame with samples and their predicted D10 values
    save_path : str, optional
        Path to save the visualization
    """
    print("Creating bar plot of predicted D10 values...")
    # Automatically adjust image size based on number of samples
    n_samples = len(predictions)
    width = max(8, n_samples * 0.3)  # Each sample requires at least 0.3 inches width
    height = 8  # Keep height fixed
    plt.figure(figsize=(width, height))
    
    # Sort predictions by D10 value for better visualization
    sorted_data = predictions.sort_values('Predicted_D10', ascending=False)
    
    # Calculate appropriate bar width based on number of samples
    bar_width = min(0.7, 5.0/len(sorted_data))  # Auto-adjust width based on sample count
    
    # Create bar plot with color coding - using lighter colors
    bars = plt.bar(
        sorted_data['Sample'], 
        sorted_data['Predicted_D10'],
        color=[('#FF8080' if val >= 1 else '#8080FF') for val in sorted_data['Predicted_D10']],  # Lighter red and blue
        width=bar_width,
        alpha=0.8  # Slightly transparent
    )
    
    # Add horizontal reference line at y=1 - now dashed and lighter
    plt.axhline(y=1, color='#FF9999', linestyle='--', linewidth=1, alpha=0.6)
    plt.text(len(sorted_data) * 0.01, 1.05, 'Reference line (D₁₀ = 1 kGy)', color='#FF6666', fontsize=10)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'{height:.2f}', ha='center', va='bottom', rotation=0)
    
    plt.title('Predicted D₁₀ values for bacterial samples', fontsize=16)
    plt.xlabel('Bacterial sample', fontsize=12)
    plt.ylabel('Predicted D₁₀ values (kGy)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add legend with lighter colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF8080', label='D₁₀ values ≥ 1 kGy (Radiation resistance bacteria)'),
        Patch(facecolor='#8080FF', label='D₁₀ values < 1 kGy (Radiation sensitive bacteria)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction visualization saved to {save_path}")
    
    # Close the plot instead of showing it
    plt.close()

def compare_transformations(predictions, save_path=None):
    """
    Create scatter plot comparing transformed and original predictions
    
    Parameters:
    -----------
    predictions : pandas.DataFrame
        DataFrame with transformed and original predictions
    save_path : str, optional
        Path to save the visualization
    """
    print("Creating comparison scatter plot of transformed vs original values...")
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(predictions['Transformed_D10'], predictions['Predicted_D10'])
    
    # Add labels for each point
    for i, sample in enumerate(predictions['Sample']):
        plt.annotate(sample, 
                    (predictions['Transformed_D10'].iloc[i], predictions['Predicted_D10'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add line of best fit
    z = np.polyfit(predictions['Transformed_D10'], predictions['Predicted_D10'], 1)
    p = np.poly1d(z)
    plt.plot(predictions['Transformed_D10'], p(predictions['Transformed_D10']), "r--", 
             label=f"Trend line: y={z[0]:.2f}x + {z[1]:.2f}")
    
    plt.title('Comparison of transformed vs original scale predictions', fontsize=16)
    plt.xlabel('Transformed D₁₀ values', fontsize=12)
    plt.ylabel('Original D₁₀ values (kGy)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Transformation comparison saved to {save_path}")
    
    # Close the plot instead of showing it
    plt.close()

def visualize_distribution(predictions, save_path=None):
    """
    Create histogram showing the distribution of predictions in both scales
    
    Parameters:
    -----------
    predictions : pandas.DataFrame
        DataFrame with transformed and original predictions
    save_path : str, optional
        Path to save the visualization
    """
    print("Creating histograms of predicted value distributions...")
    plt.figure(figsize=(12, 6))
    
    # Create two subplots
    plt.subplot(1, 2, 1)
    plt.hist(predictions['Transformed_D10'], bins=10, alpha=0.7)
    plt.title('Distribution of transformed D₁₀ values', fontsize=14)
    plt.xlabel('Transformed D₁₀ values', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(predictions['Predicted_D10'], bins=10, alpha=0.7)
    plt.title('Distribution of Original D10 Values', fontsize=14)
    plt.xlabel('D10 (kGy)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution comparison saved to {save_path}")
    
    # Close the plot instead of showing it
    plt.close()

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Predict bacterial radiation resistance from feature presence/absence matrix")
    parser.add_argument("-i", "--input", required=True, help="Input CSV file containing feature presence/absence matrix")
    parser.add_argument("-o", "--output", required=True, help="Output directory for prediction results")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        sys.exit(1)
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' does not exist")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create timestamp for file naming (moved inside main function)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Load validation data
    print(f"[1/5] Loading input data from {args.input}...")
    validation_data = pd.read_csv(args.input)
    print(f"Loaded data with shape: {validation_data.shape}")
    
    # 2. Load the trained model
    print(f"[2/5] Loading prediction model...")
    model = load_model(MODEL_PATH)
    
    if model is None:
        # If pickle loading fails, try loading from config file and recreate model
        if os.path.exists(MODEL_CONFIG_PATH):
            print(f"Attempting to load model configuration from {MODEL_CONFIG_PATH}...")
            config = load_model_config(MODEL_CONFIG_PATH)
            
            if config is None:
                print("Failed to load model. Exiting.")
                sys.exit(1)
            
            # Extract parameters and recreate model
            model_params = {k: v for k, v in config.items() if k != 'feature_importances' and k != 'best_iteration'}
            model = XGBRegressor(**model_params)
            
            # If best_iteration is available, set it
            if 'best_iteration' in config:
                model.best_iteration = config['best_iteration']
                print(f"Model recreated with best_iteration = {config['best_iteration']}")
            else:
                print("Model recreated without best_iteration.")
        else:
            print("Failed to load model and no configuration file found. Exiting.")
            sys.exit(1)
    
    # 3. Extract feature order from model configuration
    print("[3/5] Preparing features for prediction...")
    feature_order = list(validation_data.columns)
    feature_order.remove('Sample')  # Remove Sample column
    print(f"Using {len(feature_order)} features for prediction")
    
    # 4. Create D10 transformer for inverse transformation
    transformer = D10Transformer()
    print("D10 transformer initialized for inverse transformation")
    
    # 5. Make predictions
    print("[4/5] Generating predictions...")
    try:
        predictions = predict_d10_values(model, validation_data, transformer, feature_order)
        print("Predictions successfully generated")
        
        # 6. Save predictions to CSV
        print("[5/5] Saving results and creating visualizations...")
        # Only save sample names and predicted values in original scale
        predictions_to_save = predictions[['Sample', 'Predicted_D10']]
        predictions_path = os.path.join(args.output, f"predicted_d10_values_{timestamp}.csv")
        predictions_to_save.to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")
        
        # 7. Create visualization - Only keep predicted visualization
        # Visualize original scale predictions with color-coded bars
        viz_path = os.path.join(args.output, f"predicted_d10_visualization_{timestamp}.png")
        visualize_predictions(predictions, save_path=viz_path)
        
        # Output statistics to console, but do not generate files
        print("\nPrediction Summary Statistics (Original Scale):")
        summary = predictions['Predicted_D10'].describe()
        print(summary)
        
        print("\n=== Processing completed successfully ===")
        print(f"All results have been saved to: {args.output}")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()