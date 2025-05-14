# Data wrangling
import pandas as pd

# Scientific
import numpy as np

# For cloning objects
import copy

from xgboost import XGBRegressor
    
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import pickle
import os
import shap
import json
import sys
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Image, HTML

# Set random seed for reproducibility
seed = 100
np.random.seed(seed)

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Create results directory - using a relative path
results_dir = os.path.join(SCRIPT_DIR, "model_results")
os.makedirs(results_dir, exist_ok=True)

# Create timestamp for file naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Read data - using command line argument or default to example data
input_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "Example/00/results/diamond_results/presence_matrix_20250513_200601.csv")
df = pd.read_csv(input_file)
# Split features and target
X, y = df.loc[:, df.columns != 'D10'], df['D10']

# For saving model configuration as text
def save_model_config(model, filepath, avg_importance=None):
    """Save model configuration to text file"""
    # Get model parameters
    params = model.get_params()
    
    # Add feature importances if available
    if avg_importance is not None:
        feature_importance_dict = dict(zip(X.columns.tolist(), avg_importance.tolist()))
        params['feature_importances'] = feature_importance_dict
    elif hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(X.columns.tolist(), model.feature_importances_.tolist()))
        params['feature_importances'] = feature_importance
    
    # Add best iteration if available
    if hasattr(model, 'best_iteration'):
        params['best_iteration'] = model.best_iteration
    
    # Save as JSON
    with open(filepath, 'w') as file:
        json.dump(params, file, indent=4)
    
    print(f"Model configuration saved to: {filepath}")
    
    # Also save as pickle for later use
    pkl_filepath = filepath.replace('.txt', '.pkl')
    with open(pkl_filepath, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model binary saved to: {pkl_filepath}")

# For loading model
def load_model(filepath):
    """Load model from specified path"""
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model

def repeated_k_fold_cross_validation(X, y, model, n_splits=10, n_repeats=10, early_stopping_rounds=50):
    """
    Perform repeated k-fold cross-validation
    """
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scores_RMSE_test = []
    scores_MAE_test = []
    r2_test = []
    scores_RMSE_train = []
    scores_MAE_train = []
    r2_train = []
    models = []  # Save models
    rmse_scores = []  # Save RMSE scores
    evals_results = []  # Save training process
    all_feature_importances = []
    
    for train_index, test_index in cv.split(X):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        
        # Create a new model instance each time
        if hasattr(model, 'get_params'):
            model_params = model.get_params()
            # Create new model
            model_clone = XGBRegressor(**model_params)
        else:
            # Use deep copy if parameters can't be obtained
            model_clone = copy.deepcopy(model)
        
        # Add early stopping mechanism - avoid specifying eval_metric again
        model_clone.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_train_fold, y_train_fold), (X_test_fold, y_test_fold)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        
        # Save training results
        evals_results.append(model_clone.evals_result())
        
        # Predict on test set
        pred = model_clone.predict(X_test_fold)
        scores_MAE_test.append(mean_absolute_error(y_test_fold, pred))
        rmse_test = np.sqrt(mean_squared_error(y_test_fold, pred))
        scores_RMSE_test.append(rmse_test)
        r2_test.append(r2_score(y_test_fold, pred))

        # Evaluate on training set
        pred_train = model_clone.predict(X_train_fold)
        scores_MAE_train.append(mean_absolute_error(y_train_fold, pred_train))
        scores_RMSE_train.append(np.sqrt(mean_squared_error(y_train_fold, pred_train)))
        r2_train.append(r2_score(y_train_fold, pred_train))

        models.append(model_clone)  # Save current model
        rmse_scores.append(rmse_test)
        
        # Save feature importances
        all_feature_importances.append(model_clone.feature_importances_)

    # Calculate average feature importance across all models
    avg_feature_importance = np.mean(all_feature_importances, axis=0)
    
    return (np.mean(scores_RMSE_test), np.mean(scores_MAE_test), np.mean(r2_test),
            np.mean(scores_RMSE_train), np.mean(scores_MAE_train), np.mean(r2_train), 
            models, rmse_scores, evals_results, avg_feature_importance)

def plot_feature_importance(importance, feature_names, top_n=None, save_path=None):
    """Plot feature importance"""
    # If top_n is None, show all features
    if top_n is None:
        top_n = len(feature_names)
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(12, max(8, top_n * 0.3)))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Average Feature Importance Across All CV Models', fontsize=15)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    # Save to display later
    img_path = save_path
    plt.close()
    
    # Display the saved image
    try:
        display(Image(filename=img_path))
    except:
        print(f"Image saved to {img_path}")

def save_learning_curves_data(eval_results, csv_path):
    """Save learning curves data to CSV files"""
    # Extract data from all folds
    all_train_rmse = []
    all_val_rmse = []
    max_iterations = 0
    
    # Find the maximum number of iterations across all folds
    for result in eval_results:
        train_rmse = result['validation_0']['rmse']
        val_rmse = result['validation_1']['rmse']
        max_iterations = max(max_iterations, len(train_rmse))
    
    # Initialize dataframes to store data
    train_df = pd.DataFrame(index=range(1, max_iterations + 1))
    val_df = pd.DataFrame(index=range(1, max_iterations + 1))
    
    # Fill dataframes with data from each fold
    for i, result in enumerate(eval_results):
        train_rmse = result['validation_0']['rmse']
        val_rmse = result['validation_1']['rmse']
        
        # Pad with NaN if necessary
        train_series = pd.Series(train_rmse, index=range(1, len(train_rmse) + 1))
        val_series = pd.Series(val_rmse, index=range(1, len(val_rmse) + 1))
        
        train_df[f'fold_{i+1}'] = train_series
        val_df[f'fold_{i+1}'] = val_series
    
    # Calculate statistics
    train_df['mean'] = train_df.mean(axis=1)
    train_df['std'] = train_df.std(axis=1)
    train_df['min'] = train_df.min(axis=1)
    train_df['max'] = train_df.max(axis=1)
    
    val_df['mean'] = val_df.mean(axis=1)
    val_df['std'] = val_df.std(axis=1)
    val_df['min'] = val_df.min(axis=1)
    val_df['max'] = val_df.max(axis=1)
    
    # Create a combined dataframe
    combined_df = pd.DataFrame(index=range(1, max_iterations + 1))
    combined_df['train_mean'] = train_df['mean']
    combined_df['train_std'] = train_df['std']
    combined_df['train_min'] = train_df['min']
    combined_df['train_max'] = train_df['max']
    combined_df['val_mean'] = val_df['mean']
    combined_df['val_std'] = val_df['std']
    combined_df['val_min'] = val_df['min']
    combined_df['val_max'] = val_df['max']
    
    # Save to CSV
    combined_df.to_csv(csv_path)
    train_df.to_csv(csv_path.replace('.csv', '_train_detail.csv'))
    val_df.to_csv(csv_path.replace('.csv', '_val_detail.csv'))
    
    print(f"Learning curves data saved to:\n- {csv_path}\n- {csv_path.replace('.csv', '_train_detail.csv')}\n- {csv_path.replace('.csv', '_val_detail.csv')}")
    
    return combined_df

def plot_learning_curves(eval_results, save_path=None):
    """Plot learning curves"""
    # Use the first fold results for the plot
    results = eval_results[0]
    train_rmse = results['validation_0']['rmse']
    val_rmse = results['validation_1']['rmse']
    
    # Plot curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_rmse) + 1), train_rmse, 'b-', label='Training Set RMSE')
    plt.plot(range(1, len(val_rmse) + 1), val_rmse, 'r-', label='Validation Set RMSE')
    plt.title('XGBoost Model Training and Validation RMSE Curves', fontsize=15)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves plot saved to {save_path}")
    
    # Save to display later
    img_path = save_path
    plt.close()
    
    # Display the saved image
    try:
        display(Image(filename=img_path))
    except:
        print(f"Image saved to {img_path}")

def create_comprehensive_shap_analysis(models, X, results_dir, timestamp):
    """Create SHAP plots based on average predictions from all models"""
    
    print("Performing comprehensive SHAP analysis...")
    feature_count = X.shape[1]  # Get number of features
    
    # Create a SHAP explainer using the median model
    # Find median model based on RMSE
    median_idx = len(models) // 2
    median_model = models[median_idx]
    
    explainer = shap.Explainer(median_model)
    shap_values = explainer(X)
    
    # Save SHAP values as CSV
    shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
    shap_df['base_value'] = shap_values.base_values[0]  # All base values are the same
    shap_df.to_csv(f"{results_dir}/shap_values_{timestamp}.csv", index=False)
    print(f"SHAP values saved to {results_dir}/shap_values_{timestamp}.csv")
    
    # 1. Beeswarm plot - show all features
    plt.figure(figsize=(12, max(8, feature_count * 0.3)))
    shap.plots.beeswarm(shap_values, max_display=feature_count, show=False)
    plt.title('Comprehensive SHAP Feature Impact Analysis', fontsize=16)
    plt.tight_layout()
    save_path = f"{results_dir}/comprehensive_shap_beeswarm_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"SHAP beeswarm plot saved to {save_path}")
    plt.close()
    
    # Display the beeswarm plot
    try:
        display(Image(filename=save_path))
    except:
        print(f"Image saved to {save_path}")
    
    # 2. Bar plot summary - show all features
    plt.figure(figsize=(12, max(8, feature_count * 0.3)))
    shap.plots.bar(shap_values, max_display=feature_count, show=False)
    plt.title('Comprehensive SHAP Feature Importance', fontsize=16)
    plt.tight_layout()
    save_path = f"{results_dir}/comprehensive_shap_summary_{timestamp}.png" 
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"SHAP bar plot saved to {save_path}")
    plt.close()
    
    # Display the bar plot
    try:
        display(Image(filename=save_path))
    except:
        print(f"Image saved to {save_path}")

    # 3. Waterfall plot for a sample instance
    plt.figure(figsize=(12, max(8, feature_count * 0.3)))
    shap.plots.waterfall(shap_values[0], max_display=feature_count, show=False)
    plt.title('Comprehensive SHAP Waterfall Plot (Sample Instance)', fontsize=16)
    plt.tight_layout()
    save_path = f"{results_dir}/comprehensive_shap_waterfall_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"SHAP waterfall plot saved to {save_path}")
    plt.close()
    
    # Display the waterfall plot
    try:
        display(Image(filename=save_path))
    except:
        print(f"Image saved to {save_path}")
    
    # 4. Summary dot plot
    plt.figure(figsize=(12, max(8, feature_count * 0.3)))
    shap.summary_plot(shap_values.values, X, show=False)
    plt.title('Comprehensive SHAP Summary Plot', fontsize=16)
    plt.tight_layout()
    save_path = f"{results_dir}/comprehensive_shap_dot_summary_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"SHAP dot summary plot saved to {save_path}")
    plt.close()
    
    # Display the dot summary plot
    try:
        display(Image(filename=save_path))
    except:
        print(f"Image saved to {save_path}")
    
    # Export SHAP summary statistics
    shap_mean = np.abs(shap_values.values).mean(0)
    shap_summary = pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap_value': shap_mean
    }).sort_values('mean_abs_shap_value', ascending=False)
    
    shap_summary.to_csv(f"{results_dir}/shap_summary_{timestamp}.csv", index=False)
    print(f"SHAP summary statistics saved to {results_dir}/shap_summary_{timestamp}.csv")
    
    return shap_summary

# Main function
if __name__ == "__main__":
    # Use the exact parameters from the JSON configuration
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=1,
        gamma=0,
        subsample=0.26,
        colsample_bytree=0.5,
        reg_alpha=0,
        reg_lambda=1,
        objective='reg:squarederror',
        eval_metric='rmse',
        n_jobs=-1,
        random_state=seed
    )
    
    # Perform cross-validation
    print("Performing cross-validation...")
    results = repeated_k_fold_cross_validation(
        X, y, model, n_splits=10, n_repeats=10, early_stopping_rounds=50
    )
    
    scores_RMSE_test, scores_MAE_test, r2_test, scores_RMSE_train, scores_MAE_train, r2_train, models, rmse_scores, evals_results, avg_importance = results
    
    # Output results
    print("Cross-validation Results:")
    print(f"Test set RMSE: {scores_RMSE_test:.6f}, MAE: {scores_MAE_test:.6f}, R²: {r2_test:.6f}")
    print(f"Training set RMSE: {scores_RMSE_train:.6f}, MAE: {scores_MAE_train:.6f}, R²: {r2_train:.6f}")
    
    # Find median model (most representative)
    sorted_indices = np.argsort(rmse_scores)
    median_idx = len(sorted_indices) // 2
    median_model = models[sorted_indices[median_idx]]
    median_rmse = rmse_scores[sorted_indices[median_idx]]
    
    print(f"Median model RMSE: {median_rmse:.6f}")
    print(f"RMSE range: {min(rmse_scores):.6f} - {max(rmse_scores):.6f}")
    
    # Save median model as TEXT and PKL with average importance
    save_model_config(
        median_model, 
        f"{results_dir}/comprehensive_model_{timestamp}.txt",
        avg_importance=avg_importance
    )
    
    # Export cross-validation results to CSV
    cv_results = pd.DataFrame({
        'model_index': range(len(rmse_scores)),
        'rmse': rmse_scores
    })
    cv_results.to_csv(f"{results_dir}/cv_results_{timestamp}.csv", index=False)
    print(f"Cross-validation results saved to {results_dir}/cv_results_{timestamp}.csv")
    
    # Create comprehensive SHAP analysis
    shap_summary = create_comprehensive_shap_analysis(models, X, results_dir, timestamp)
    
    # Export learning curves data to CSV
    learning_curves_csv = f"{results_dir}/learning_curves_{timestamp}.csv"
    learning_curves_df = save_learning_curves_data(evals_results, learning_curves_csv)
    
    # Plot learning curves 
    learning_curves_path = f"{results_dir}/learning_curves_{timestamp}.png"
    plot_learning_curves(evals_results, save_path=learning_curves_path)
    
    # Plot average feature importance - show ALL features
    feature_importance_path = f"{results_dir}/avg_feature_importance_{timestamp}.png"
    plot_feature_importance(avg_importance, X.columns, top_n=None, save_path=feature_importance_path)
    
    # Feature importance analysis - Sort and output importance of all features
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': avg_importance
    }).sort_values('importance', ascending=False)
    
    print("\nAverage Feature Importance Ranking:")
    print(feature_importance)
    feature_importance.to_csv(f"{results_dir}/avg_feature_importance_{timestamp}.csv", index=False)
    
    # Export model summary statistics
    model_summary = pd.DataFrame({
        'metric': ['RMSE_test', 'MAE_test', 'R2_test', 
                  'RMSE_train', 'MAE_train', 'R2_train',
                  'RMSE_min', 'RMSE_max', 'RMSE_median'],
        'value': [scores_RMSE_test, scores_MAE_test, r2_test,
                 scores_RMSE_train, scores_MAE_train, r2_train,
                 min(rmse_scores), max(rmse_scores), median_rmse]
    })
    model_summary.to_csv(f"{results_dir}/model_summary_{timestamp}.csv", index=False)
    print(f"Model summary statistics saved to {results_dir}/model_summary_{timestamp}.csv")
    
    print(f"\nAll results have been saved to the '{results_dir}' directory.")
    
    # Display a summary of the most important features
    top_features = feature_importance.head(10)
    display(HTML("<b>Top 10 Most Important Features:</b>"))
    display(top_features)
    
    # Display model performance summary
    display(HTML("<b>Model Performance Summary:</b>"))
    display(model_summary)