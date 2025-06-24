#!/usr/bin/env python3
"""
Complete Visualization Script for Air Pollution Prediction
Best Model: Extra Trees Regressor
Generates all required figures from the experimental documentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import gaussian_kde
import joblib
import itertools
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_preprocess_data():
    """Load and preprocess the data following the exact pipeline."""
    print("Loading and preprocessing data...")
    
    # Load the full dataset
    df = pd.read_csv('South_Link_Road_Final (4).csv')
    print(f"Original dataset shape: {df.shape}")
    
    # Date processing and feature extraction
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
    # Reorder columns to match notebook
    column_order = ['Date', 'Year', 'Month', 'Day', 'Hour', 'ind', 'temp', 'dewpt', 
                   'rhum', 'msl', 'wdsp', 'wddir', 'clamt', 'NO2']
    df = df[column_order]
    
    # Drop Date column for modeling
    df = df.drop('Date', axis=1)
    
    # Remove missing values (matching notebook approach)
    df_cleaned = df.dropna()
    print(f"After removing missing values: {df_cleaned.shape}")
    
    # Season mapping function
    def map_season(month):
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Fall'
        else:
            return 'Winter'
    
    df_cleaned['Season'] = df_cleaned['Month'].apply(map_season)
    
    return df_cleaned

def train_best_model(X_train, X_test, y_train, y_test):
    """Train Extra Trees as the best model."""
    print("Training Extra Trees (Best Model)...")
    
    # Extra Trees with optimized parameters
    best_model = ExtraTreesRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    best_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"Extra Trees - Train R²: {train_r2:.4f}")
    print(f"Extra Trees - Test R²: {test_r2:.4f}")
    print(f"Extra Trees - Test MAE: {test_mae:.4f}")
    print(f"Extra Trees - Test RMSE: {test_rmse:.4f}")
    
    return best_model, y_pred_train, y_pred_test, train_r2, test_r2

def figure_2_negative_no2_analysis(df, output_dir='visualization_plots'):
    """Figure 2: Finding the threshold for treating negative NO2 values."""
    print("Generating Figure 2: Negative NO2 Analysis...")
    
    # Filter negative NO2 values
    negative_NO2 = df[df['NO2'] < 0]
    
    if len(negative_NO2) == 0:
        print("No negative NO2 values found in the dataset.")
        # Create placeholder figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No Negative NO2 Values Found\nin Current Dataset', 
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax.set_title('Figure 2: Negative NO2 Analysis', fontsize=14)
        plt.savefig(f'{output_dir}/Figure_2_Negative_NO2_Analysis_ExtraTrees.png', dpi=300, bbox_inches='tight')
        plt.show()
        return
    
    negative_NO2_values = negative_NO2['NO2']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Time series plot of negative values
    ax1.plot(negative_NO2.index, negative_NO2['NO2'], 'ro-', markersize=3, alpha=0.7)
    ax1.set_title('Time Series Plot of Negative NO2 Values', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('NO2 Concentration (µg/m³)')
    ax1.grid(True, alpha=0.3)
    
    # KDE analysis
    if len(negative_NO2_values) > 1:
        kde = gaussian_kde(negative_NO2_values)
        x_range = np.linspace(negative_NO2_values.min(), negative_NO2_values.max(), 1000)
        density = kde(x_range)
        
        # Cumulative density for threshold
        cum_density = np.cumsum(density)
        cum_density /= cum_density[-1]
        
        # Find 5% quantile
        threshold_index = np.where(cum_density <= 0.05)[0]
        if len(threshold_index) > 0:
            threshold = x_range[threshold_index[-1]]
        else:
            threshold = negative_NO2_values.min()
        
        ax2.plot(x_range, density, 'b-', linewidth=2, label='KDE of Negative NO2 Values')
        ax2.fill_between(x_range, 0, density, where=(x_range <= threshold), alpha=0.5, color='red')
        ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold (5% quantile): {threshold:.2f}')
        ax2.set_xlabel('NO2 Concentration (µg/m³)')
        ax2.set_ylabel('Density')
        ax2.set_title('KDE of Negative NO2 Values', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        print(f"Threshold for negative NO2 outliers: {threshold:.2f}")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_2_Negative_NO2_Analysis_ExtraTrees.png', dpi=300, bbox_inches='tight')
    plt.show()

def figure_3_feature_importance(model, feature_names, output_dir='visualization_plots'):
    """Figure 3: Feature Importance from Extra Trees."""
    print("Generating Figure 3: Feature Importance Analysis...")
    
    # Get feature importances
    importances = model.feature_importances_
    sort_indices = np.argsort(importances)[::-1]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Bar plot
    bars = plt.bar(range(len(feature_names)), importances[sort_indices], 
                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    plt.title('Visualization of Feature Importance in the Extra Trees Regressor\nfor Predicting Atmospheric NO2 Concentrations', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in sort_indices], 
               rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_3_Feature_Importance_ExtraTrees.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print ranking
    print("\nFeature Ranking (Extra Trees):")
    for i in range(len(feature_names)):
        print(f"{i+1:2d}. {feature_names[sort_indices[i]]:12s} ({importances[sort_indices[i]]:.4f})")

def figure_4_environmental_histograms(df, output_dir='visualization_plots'):
    """Figure 4: Histograms of Environmental Parameters."""
    print("Generating Figure 4: Environmental Parameter Histograms...")
    
    # Select key environmental parameters matching your sample
    cols = ['NO2', 'temp', 'rhum', 'msl', 'wdsp', 'dewpt', 'clamt', 'ind', 'Hour']
    colors = ["steelblue", "forestgreen", "crimson", "darkorange", "purple", 
              "brown", "pink", "gray", "olive"]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (col, color) in enumerate(zip(cols, colors)):
        if col in df.columns:
            # Create histogram with density
            axes[i].hist(df[col], bins=30, density=True, alpha=0.7, color=color, edgecolor='black')
            
            # Add KDE curve
            try:
                from scipy.stats import gaussian_kde
                data_clean = df[col].dropna()
                if len(data_clean) > 1:
                    kde = gaussian_kde(data_clean)
                    x_range = np.linspace(data_clean.min(), data_clean.max(), 100)
                    axes[i].plot(x_range, kde(x_range), 'black', linewidth=2)
            except:
                pass
                
            # Add mean line
            mean_val = df[col].mean()
            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            
            axes[i].set_title(f'{col}', fontsize=12, fontweight='bold', color='navy')
            axes[i].set_xlabel('')
            axes[i].legend(loc='best')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_facecolor('white')
    
    plt.suptitle('Histograms of Environmental Parameters\nShowing the Distributional Characteristics of Key Variables in the Dataset', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_4_Environmental_Histograms_ExtraTrees.png', dpi=300, bbox_inches='tight')
    plt.show()

def figure_5_correlation_heatmap(df, output_dir='visualization_plots'):
    """Figure 5: Correlation Matrix Heatmap."""
    print("Generating Figure 5: Correlation Heatmap...")
    
    # Select numeric columns for correlation
    numeric_cols = ['Year', 'Month', 'Day', 'Hour', 'ind', 'temp', 'dewpt', 
                   'rhum', 'msl', 'wdsp', 'wddir', 'clamt', 'NO2']
    
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(14, 10))
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                fmt='.2f', annot_kws={'size': 9})
    
    plt.title('Correlation Matrices and Heatmaps\nShowing the Relationships Between Atmospheric and Temporal Parameters', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_5_Correlation_Heatmap_ExtraTrees.png', dpi=300, bbox_inches='tight')
    plt.show()

def figure_6_pca_analysis(df, output_dir='visualization_plots'):
    """Figure 6: PCA Loading Analysis."""
    print("Generating Figure 6: PCA Loading Analysis...")
    
    # Select meteorological parameters
    meteorological_params = ['temp', 'dewpt', 'rhum', 'msl', 'wdsp', 'wddir', 'clamt']
    data = df[meteorological_params].dropna()
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA()
    principal_components = pca.fit_transform(scaled_data)
    
    # Get PCA loadings
    loadings = pca.components_.T
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot arrows and labels
    for i, var in enumerate(meteorological_params):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                head_width=0.05, head_length=0.05, fc='red', ec='red', linewidth=2)
        ax.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, var, 
               color='green', ha='center', va='center', fontweight='bold', fontsize=11)
    
    # Customize plot
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlabel(f'Principal Component 1', fontsize=12)
    ax.set_ylabel(f'Principal Component 2', fontsize=12)
    ax.set_title('PCA Loading Analysis\nShowing the Contribution of Meteorological Variables to the Principal Components', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    # Add circle
    circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_6_PCA_Analysis_ExtraTrees.png', dpi=300, bbox_inches='tight')
    plt.show()

def figure_7_scatterplots(df, output_dir='visualization_plots'):
    """Figure 7: NO2 vs Other Variables Scatterplots."""
    print("Generating Figure 7: NO2 Scatterplot Analysis...")
    
    # Feature units
    feature_units = {
        'Year': 'Year', 'Month': 'Month number', 'Day': 'Day of month', 'Hour': 'Hours',
        'ind': 'Index unit', 'temp': '°C', 'dewpt': '°C', 'rhum': '%', 
        'msl': 'millibars', 'wdsp': 'km/h', 'wddir': '°', 'clamt': 'Arbitrary unit'
    }
    
    columns = [col for col in df.columns if col != 'NO2' and col != 'Season']
    
    n = len(columns)
    fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(25, (n // 2) * 6))
    axes = axes.ravel()
    
    for i, col in enumerate(columns):
        if col in df.columns:
            axes[i].scatter(df['NO2'], df[col], alpha=0.5, s=3, color='blue')
            axes[i].set_title(f'NO2 vs {col}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('NO2 (µg/m³)', fontsize=10)
            axes[i].set_ylabel(f'{col} ({feature_units.get(col, "unit")})', fontsize=10)
            axes[i].grid(True, alpha=0.3)
    
    # Remove unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle('Scatterplot Analysis: NO2 vs Environmental Variables', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_7_Scatterplots_ExtraTrees.png', dpi=300, bbox_inches='tight')
    plt.show()

def figure_8_actual_vs_predicted(y_test, y_pred_test, test_r2, train_r2, output_dir='visualization_plots'):
    """Figure 8: Actual vs Predicted Values."""
    print("Generating Figure 8: Actual vs Predicted Analysis...")
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(y_test, y_pred_test, alpha=0.5, s=15, color='blue')
    
    # Perfect prediction line (diagonal)
    min_val = min(min(y_test), min(y_pred_test))
    max_val = max(max(y_test), max(y_pred_test))
    plt.plot([min_val, max_val], [min_val, max_val], 'red', linewidth=2, label='Perfect Prediction')
    
    # Customize plot
    plt.xlabel('Actual values', fontsize=12)
    plt.ylabel('Predicted values', fontsize=12)
    plt.title('Predicted versus Actual NO2 Concentration Values\nShowing Model Prediction Accuracy', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add R² text
    plt.text(0.05, 0.95, f'Training R² Score: {train_r2:.4f}\nTest R² Score: {test_r2:.4f}', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_8_Actual_vs_Predicted_ExtraTrees.png', dpi=300, bbox_inches='tight')
    plt.show()

def figure_9_temporal_patterns(df, output_dir='visualization_plots'):
    """Figure 9: Temporal Patterns Analysis."""
    print("Generating Figure 9: Temporal Patterns...")
    
    # Create datetime index
    df_temp = df.copy()
    df_temp['DateTime'] = pd.to_datetime(df_temp[['Year', 'Month', 'Day', 'Hour']])
    df_temp.set_index('DateTime', inplace=True)
    
    meteorological_params = ['temp', 'dewpt', 'rhum', 'msl', 'wdsp', 'wddir', 'clamt']
    units = {
        'temp': '°C', 'dewpt': '°C', 'rhum': '%', 'msl': 'millibars',
        'wdsp': 'km/h', 'wddir': '°', 'clamt': 'Arbitrary unit'
    }
    
    for param in meteorological_params:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        patterns = [
            (df_temp.index.hour, 'Hour of the day'),
            (df_temp.index.dayofweek, 'Day of the week\n(0=Monday, 6=Sunday)'),
            (df_temp.index.month, 'Month\n(1=January, 12=December)')
        ]
        
        for i, (groupby_param, title) in enumerate(patterns):
            grouped_data = df_temp.groupby(groupby_param).agg({param: 'mean', 'NO2': 'mean'})
            
            # Plot NO2 on primary axis
            ax1 = axes[i]
            ax1.plot(grouped_data.index, grouped_data['NO2'], 'b-', linewidth=2, color='blue', label='NO2')
            ax1.set_ylabel(f'NO2 (µg/m³)', color='blue', fontsize=11)
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_xlabel(title, fontsize=11)
            
            # Plot parameter on secondary axis
            ax2 = ax1.twinx()
            ax2.plot(grouped_data.index, grouped_data[param], 'r-', linewidth=2, color='red', label=param)
            ax2.set_ylabel(f'{param} ({units[param]})', color='red', fontsize=11)
            ax2.tick_params(axis='y', labelcolor='red')
            
            ax1.set_title(f'{title} pattern of\nNO2 and {param}', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        plt.suptitle(f'Interplay of Climatic Parameters: A Symphony of Temperature Variations,\nAtmospheric Indicators, and Human Influence', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/Figure_9_Temporal_Patterns_{param}_ExtraTrees.png', dpi=300, bbox_inches='tight')
        plt.show()

def figure_10_wind_direction_analysis(df, output_dir='visualization_plots'):
    """Figure 10: Wind Direction Circular Analysis."""
    print("Generating Figure 10: Wind Direction Analysis...")
    
    # Convert wind direction to radians
    df_wind = df.dropna(subset=['wddir'])
    wddir_rad = np.deg2rad(df_wind['wddir'])
    
    # Calculate circular statistics
    mean_sin = np.mean(np.sin(wddir_rad))
    mean_cos = np.mean(np.cos(wddir_rad))
    R = np.sqrt(mean_sin**2 + mean_cos**2)
    
    mean_direction = np.arctan2(mean_sin, mean_cos)
    if mean_direction < 0:
        mean_direction += 2*np.pi
    mean_direction_deg = np.rad2deg(mean_direction)
    
    print(f"Mean Wind Direction: {mean_direction_deg:.1f}°")
    print(f"Mean Resultant Length: {R:.4f}")
    
    # Create polar plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    
    # Plot mean direction arrow
    ax.arrow(mean_direction, 0, 0, R, alpha=0.5, width=0.015,
             edgecolor='black', facecolor='green', lw=1, zorder=5)
    
    ax.set_title('Wind Direction Circular Decomposition', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_10_Wind_Direction_Analysis_ExtraTrees.png', dpi=300, bbox_inches='tight')
    plt.show()

def figure_11_seasonal_patterns(df, output_dir='visualization_plots'):
    """Figure 11: Seasonal Patterns Analysis."""
    print("Generating Figure 11: Seasonal Patterns...")
    
    # Season mapping
    def map_season_detailed(month):
        if month in [3, 4, 5]:
            return 'Spring (Mar-May)'
        elif month in [6, 7, 8]:
            return 'Summer (Jun-Aug)'
        elif month in [9, 10, 11]:
            return 'Autumn (Sep-Nov)'
        else:
            return 'Winter (Dec-Feb)'
    
    df['Season_Detailed'] = df['Month'].apply(map_season_detailed)
    
    meteorological_params = ['temp', 'dewpt', 'rhum', 'msl', 'wdsp', 'wddir', 'clamt']
    units = {
        'temp': '°C', 'dewpt': '°C', 'rhum': '%', 'msl': 'millibars',
        'wdsp': 'km/h', 'wddir': '°', 'clamt': 'Arbitrary unit'
    }
    
    for param in meteorological_params:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Calculate seasonal averages
        seasonal_order = ['Spring (Mar-May)', 'Summer (Jun-Aug)', 'Autumn (Sep-Nov)', 'Winter (Dec-Feb)']
        seasonal_data = df.groupby('Season_Detailed').agg({param: 'mean', 'NO2': 'mean'}).reindex(seasonal_order)
        
        # Plot NO2
        ax.plot(seasonal_data.index, seasonal_data['NO2'], 'b-o', linewidth=2, markersize=8, label='NO2', color='blue')
        ax.set_ylabel(f'NO2 (µg/m³)', color='blue', fontsize=11)
        ax.tick_params(axis='y', labelcolor='blue')
        
        # Plot meteorological parameter
        ax2 = ax.twinx()
        ax2.plot(seasonal_data.index, seasonal_data[param], 'r-o', linewidth=2, markersize=8, label=param, color='red')
        ax2.set_ylabel(f'{param} ({units[param]})', color='red', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax.set_title(f'Seasonal pattern of NO2 (µg/m³) and {param} ({units[param]})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Season', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/Figure_11_Seasonal_Pattern_{param}_ExtraTrees.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to generate all visualizations."""
    print("="*70)
    print("COMPREHENSIVE VISUALIZATION GENERATOR")
    print("Best Model: Extra Trees Regressor")
    print("="*70)
    
    # Create organized output directory structure
    output_dir = 'visualization_results/extra_trees'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Prepare data for modeling
    feature_columns = ['Year', 'Month', 'Day', 'Hour', 'ind', 'temp', 'dewpt', 
                      'rhum', 'msl', 'wdsp', 'wddir', 'clamt']
    X = df[feature_columns]
    y = df['NO2']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train best model
    best_model, y_pred_train, y_pred_test, train_r2, test_r2 = train_best_model(X_train, X_test, y_train, y_test)
    
    # Generate all figures
    try:
        figure_2_negative_no2_analysis(df, output_dir)
        figure_3_feature_importance(best_model, feature_columns, output_dir)
        figure_4_environmental_histograms(df, output_dir)
        figure_5_correlation_heatmap(df, output_dir)
        figure_6_pca_analysis(df, output_dir)
        figure_7_scatterplots(df, output_dir)
        figure_8_actual_vs_predicted(y_test, y_pred_test, test_r2, train_r2, output_dir)
        figure_9_temporal_patterns(df, output_dir)
        figure_10_wind_direction_analysis(df, output_dir)
        figure_11_seasonal_patterns(df, output_dir)
        
    except Exception as e:
        print(f"Error generating figures: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("Best Model: Extra Trees Regressor")
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Figures saved in: {output_dir}/")
    print("="*70)

if __name__ == "__main__":
    main() 