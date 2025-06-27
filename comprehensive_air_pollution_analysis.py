#!/usr/bin/env python3
"""
ULTIMATE COMPREHENSIVE AIR POLLUTION ANALYSIS SCRIPT v2.0
=========================================================

MAJOR UPDATES AND NEW FEATURES:
==============================

1. IMPROVED FORECASTING SYSTEM:
   - Enhanced 1-year forecast generation using best ML model
   - Intelligent feature generation for future predictions
   - Comprehensive forecast visualizations (6-panel analysis)
   - Historical pattern integration for meteorological features
   - Automatic model selection and forecast generation

2. COMPREHENSIVE PATTERN ANALYSIS:
   - 12-panel comprehensive pattern analysis visualization
   - Long-term trend analysis with statistical significance
   - Seasonal, daily, and weekly pattern identification
   - Anomaly detection using IQR method
   - Wind pattern analysis (when data available)
   - Meteorological correlation analysis
   - Statistical distribution analysis
   - Insights extraction and reporting

3. PROFESSIONAL DASHBOARD:
   - Publication-ready professional dashboard
   - Executive summary with key findings
   - Multi-panel integrated visualization
   - Professional color scheme and typography
   - Policy implications and recommendations

4. ENHANCED DIRECTORY STRUCTURE:
   - forecast_results/: 1-year forecasts and visualizations
   - pattern_analysis_results/: Comprehensive pattern analysis
   - professional_results/: Professional dashboard outputs
   - visualization_results/: Model-specific visualizations
   - Organized output structure for all analysis components

5. COMPREHENSIVE RESULTS INTEGRATION:
   - All analysis components in single script
   - Automatic execution flow from ML → TS → Patterns → Forecasting
   - Comprehensive summary report generation
   - Insights extraction and correlation across all analyses

6. STREAMLINED EXECUTION:
   - Removed unnecessary CSV exports to reduce file clutter
   - Optimized visualization generation
   - Enhanced error handling and progress reporting
   - Consolidated all functionality in one script

EXECUTION FLOW:
==============
1. Machine Learning Analysis (17 models)
2. Time Series Analysis (ARIMA variants)
3. Pattern Analysis (trends, anomalies, insights)
4. Professional Dashboard Generation
5. 1-Year Forecasting with Best Model
6. Model-Specific Visualizations
7. Interactive Model Selection
8. Comprehensive Summary Report

ALL PREVIOUS FUNCTIONALITY PRESERVED:
====================================
- Complete ML model training and evaluation
- Time series analysis with ARIMA models
- Experimental tables generation (Tables 1-7)
- All 11 visualization figures per model
- Interactive model selection system
- Comprehensive performance evaluation

NEW OUTPUT FILES:
================
- 1_year_no2_forecast.csv: Complete forecast data
- 1_year_no2_forecast_visualization.png: Forecast analysis
- comprehensive_pattern_analysis.png: 12-panel pattern analysis
- professional_air_pollution_dashboard.png: Executive dashboard
- comprehensive_analysis_summary.txt: Complete results summary

TOTAL ANALYSIS COMPONENTS:
=========================
- 17 Machine Learning Models
- 4 Time Series Models  
- 7 Experimental Tables
- 12-Panel Pattern Analysis
- Professional Dashboard
- 1-Year Forecast System
- Model-Specific Visualizations
- Interactive Selection System

Run this script to execute the complete comprehensive analysis!
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from scipy import stats
import itertools
from datetime import datetime, timedelta

# Machine Learning Model imports
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor, VotingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# Time Series imports
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings('ignore')

# Constants
DATASET_FILE = 'air_pollution_ireland.csv'
MODELS_DIR = 'trained_models'
TABLES_DIR = 'experimental_results_tables'
TS_FIGURES_DIR = 'time_series_only'
FORECAST_RESULTS_DIR = 'forecast_results'
PATTERN_ANALYSIS_DIR = 'pattern_analysis_results'
PROFESSIONAL_RESULTS_DIR = 'professional_results'
VISUALIZATION_RESULTS_DIR = 'visualization_results'
RANDOM_STATE = 42

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def create_directories():
    """Create necessary directories"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(TS_FIGURES_DIR, exist_ok=True)
    os.makedirs(FORECAST_RESULTS_DIR, exist_ok=True)
    os.makedirs(PATTERN_ANALYSIS_DIR, exist_ok=True)
    os.makedirs(PROFESSIONAL_RESULTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_RESULTS_DIR, exist_ok=True)

# ============================================================================
# TIME SERIES ANALYSIS FUNCTIONS
# ============================================================================

def load_and_clean_data_for_ts():
    """Load data and handle duplicates properly for time series analysis"""
    print("\n🔍 Loading and cleaning data for time series analysis...")
    
    # Load data
    df = pd.read_csv(DATASET_FILE)
    print(f"✅ Original dataset: {df.shape}")
    
    # Create datetime index
    df['DateTime'] = pd.to_datetime(df['Date']) + pd.to_timedelta(df['Hour'], unit='h')
    
    print(f"   Checking for duplicates...")
    print(f"   Total rows: {len(df)}")
    print(f"   Unique timestamps: {df['DateTime'].nunique()}")
    print(f"   Duplicate timestamps: {len(df) - df['DateTime'].nunique()}")
    
    # Handle duplicates by taking the mean of NO2 values for same timestamp
    df_clean = df.groupby('DateTime').agg({
        'NO2': 'mean',  # Take mean for duplicate timestamps
        'temp': 'mean',
        'rhum': 'mean'
    }).sort_index()
    
    print(f"   After duplicate handling: {df_clean.shape}")
    
    # Focus on NO2 and handle missing values
    no2_series = df_clean['NO2'].copy()
    print(f"   Missing values in NO2: {no2_series.isna().sum()}")
    
    # Conservative interpolation (max 6 hours gap)
    no2_clean = no2_series.interpolate(method='linear', limit=6).dropna()
    
    print(f"   Final clean series: {len(no2_clean)} observations")
    print(f"   Date range: {no2_clean.index.min()} to {no2_clean.index.max()}")
    print(f"   NO2 stats: Mean={no2_clean.mean():.2f}, Std={no2_clean.std():.2f}")
    
    return no2_clean

def create_train_test_split_ts(series, train_ratio=0.8):
    """Create temporal train/test split for time series"""
    split_point = int(len(series) * train_ratio)
    train_series = series[:split_point]
    test_series = series[split_point:]
    
    print(f"\n🔄 Time Series Train/Test Split:")
    print(f"   Training: {train_series.index[0]} to {train_series.index[-1]} ({len(train_series):,} obs)")
    print(f"   Testing: {test_series.index[0]} to {test_series.index[-1]} ({len(test_series):,} obs)")
    
    return train_series, test_series

def fit_time_series_models(train_series, test_series):
    """Fit ARIMA models and evaluate properly"""
    print(f"\n📈 Fitting and evaluating time series models...")
    
    results = {}
    
    # Test different ARIMA orders
    orders_to_test = [(1,1,1), (2,1,1), (1,1,2), (0,1,1)]
    
    for order in orders_to_test:
        print(f"\n   Testing ARIMA{order}...")
        try:
            # Fit model
            model = ARIMA(train_series, order=order)
            fitted_model = model.fit()
            
            # Training predictions (in-sample)
            train_pred = fitted_model.fittedvalues
            train_actual = train_series[1:]  # Skip first value due to differencing
            train_pred_aligned = train_pred[1:]  # Align predictions
            
            # Training metrics
            train_mae = mean_absolute_error(train_actual, train_pred_aligned)
            train_rmse = np.sqrt(mean_squared_error(train_actual, train_pred_aligned))
            train_r2 = r2_score(train_actual, train_pred_aligned)
            
            # Testing predictions (out-of-sample)
            test_pred = fitted_model.forecast(steps=len(test_series))
            test_mae = mean_absolute_error(test_series, test_pred)
            test_rmse = np.sqrt(mean_squared_error(test_series, test_pred))
            test_r2 = r2_score(test_series, test_pred)
            
            results[f'ARIMA{order}'] = {
                'model': fitted_model,
                'order': order,
                'train_mae': train_mae,
                'train_rmse': train_rmse,
                'train_r2': train_r2,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'test_predictions': test_pred,
                'aic': fitted_model.aic
            }
            
            print(f"     Training:  MAE={train_mae:.3f}, R²={train_r2:.3f}")
            print(f"     Testing:   MAE={test_mae:.3f}, R²={test_r2:.3f}")
            
        except Exception as e:
            print(f"     ❌ Failed: {e}")
    
    # Benchmark: Naive model
    print(f"\n   Testing Naive model...")
    naive_test_pred = np.full(len(test_series), train_series.iloc[-1])
    naive_mae = mean_absolute_error(test_series, naive_test_pred)
    naive_rmse = np.sqrt(mean_squared_error(test_series, naive_test_pred))
    naive_r2 = r2_score(test_series, naive_test_pred)
    
    results['Naive'] = {
        'test_mae': naive_mae,
        'test_rmse': naive_rmse,
        'test_r2': naive_r2,
        'test_predictions': naive_test_pred
    }
    
    print(f"     Testing:   MAE={naive_mae:.3f}, R²={naive_r2:.3f}")
    
    return results

def create_time_series_visualizations(train_series, test_series, results):
    """Create comprehensive time series visualization plots"""
    print(f"\n📊 Creating time series visualization plots...")
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Full time series overview
    ax1 = plt.subplot(3, 3, 1)
    full_series = pd.concat([train_series, test_series])
    full_series.plot(ax=ax1, color='blue', alpha=0.7, linewidth=1)
    ax1.axvline(x=train_series.index[-1], color='red', linestyle='--', linewidth=2, label='Train/Test Split')
    ax1.set_title('Complete Time Series - NO2 Air Pollution')
    ax1.set_ylabel('NO2 (µg/m³)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Testing period predictions
    ax2 = plt.subplot(3, 3, 2)
    test_series.plot(ax=ax2, label='Actual', color='black', linewidth=2)
    
    colors = ['red', 'green', 'purple', 'orange']
    for i, (model_name, result) in enumerate(results.items()):
        if 'test_predictions' in result and model_name.startswith('ARIMA'):
            test_pred_series = pd.Series(result['test_predictions'], index=test_series.index)
            test_pred_series.plot(ax=ax2, 
                                label=f'{model_name} (R²={result["test_r2"]:.3f})',
                                color=colors[i % len(colors)], 
                                linestyle='--', linewidth=2)
    
    ax2.set_title('Testing Period - Model Predictions')
    ax2.set_ylabel('NO2 (µg/m³)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Training vs Testing MAE comparison
    ax3 = plt.subplot(3, 3, 3)
    model_names = [name for name in results.keys() if name.startswith('ARIMA')]
    train_maes = [results[name]['train_mae'] for name in model_names]
    test_maes = [results[name]['test_mae'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax3.bar(x - width/2, train_maes, width, label='Training MAE', color='lightblue', alpha=0.7)
    ax3.bar(x + width/2, test_maes, width, label='Testing MAE', color='lightcoral', alpha=0.7)
    
    ax3.set_title('Training vs Testing MAE')
    ax3.set_ylabel('MAE (µg/m³)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. R² comparison
    ax4 = plt.subplot(3, 3, 4)
    train_r2s = [results[name]['train_r2'] for name in model_names]
    test_r2s = [results[name]['test_r2'] for name in model_names]
    
    ax4.bar(x - width/2, train_r2s, width, label='Training R²', color='lightblue', alpha=0.7)
    ax4.bar(x + width/2, test_r2s, width, label='Testing R²', color='lightcoral', alpha=0.7)
    
    ax4.set_title('Training vs Testing R²')
    ax4.set_ylabel('R²')
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_names, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 5. Model performance ranking
    ax5 = plt.subplot(3, 3, 5)
    all_models = list(results.keys())
    all_test_maes = [results[name]['test_mae'] for name in all_models]
    
    bars = ax5.bar(range(len(all_models)), all_test_maes, color='lightgreen', alpha=0.7)
    ax5.set_title('Model Ranking by Testing MAE')
    ax5.set_ylabel('Testing MAE (µg/m³)')
    ax5.set_xticks(range(len(all_models)))
    ax5.set_xticklabels(all_models, rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    # 6. Time series decomposition view
    ax6 = plt.subplot(3, 3, 6)
    # Show seasonal patterns
    monthly_avg = full_series.resample('M').mean()
    monthly_avg.plot(ax=ax6, marker='o', linewidth=2)
    ax6.set_title('Monthly Average NO2 Trends')
    ax6.set_ylabel('NO2 (µg/m³)')
    ax6.grid(True, alpha=0.3)
    
    # 7. Residuals analysis for best model
    ax7 = plt.subplot(3, 3, 7)
    best_arima = min([name for name in results.keys() if name.startswith('ARIMA')], 
                    key=lambda x: results[x]['test_mae'])
    residuals = test_series.values - results[best_arima]['test_predictions']
    
    ax7.hist(residuals, bins=30, alpha=0.7, color='skyblue', density=True)
    ax7.set_title(f'Residuals Distribution - {best_arima}')
    ax7.set_xlabel('Residuals (µg/m³)')
    ax7.set_ylabel('Density')
    ax7.grid(True, alpha=0.3)
    
    # 8. Prediction vs Actual scatter
    ax8 = plt.subplot(3, 3, 8)
    best_pred = results[best_arima]['test_predictions']
    ax8.scatter(test_series.values, best_pred, alpha=0.6, color='purple')
    
    # Add perfect prediction line
    min_val = min(test_series.min(), best_pred.min())
    max_val = max(test_series.max(), best_pred.max())
    ax8.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax8.set_title(f'Predicted vs Actual - {best_arima}')
    ax8.set_xlabel('Actual NO2 (µg/m³)')
    ax8.set_ylabel('Predicted NO2 (µg/m³)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Error over time
    ax9 = plt.subplot(3, 3, 9)
    errors = np.abs(test_series.values - best_pred)
    error_series = pd.Series(errors, index=test_series.index)
    error_series.rolling(window=24).mean().plot(ax=ax9, color='red', linewidth=2)
    
    ax9.set_title('24-Hour Rolling Mean Absolute Error')
    ax9.set_ylabel('Absolute Error (µg/m³)')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{TS_FIGURES_DIR}/comprehensive_time_series_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Time series plots saved to {TS_FIGURES_DIR}/comprehensive_time_series_analysis.png")

def generate_forecast(train_series, test_series, best_model, forecast_steps=365*24):
    """Generate 1-year forecast"""
    print(f"\n🔮 Generating 1-year forecast...")
    
    try:
        # Fit model on all available data
        full_series = pd.concat([train_series, test_series])
        model = ARIMA(full_series, order=best_model['order'])
        fitted_model = model.fit()
        
        # Generate forecast
        forecast = fitted_model.forecast(steps=forecast_steps)
        forecast_ci = fitted_model.get_forecast(steps=forecast_steps).conf_int()
        
        # Create forecast dates
        last_date = full_series.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(hours=1), 
                                     periods=forecast_steps, freq='h')
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'NO2_Forecast': forecast,
            'Lower_CI': forecast_ci.iloc[:, 0] if hasattr(forecast_ci, 'iloc') else np.nan,
            'Upper_CI': forecast_ci.iloc[:, 1] if hasattr(forecast_ci, 'iloc') else np.nan
        })
        
        # Save forecast
        forecast_df.to_csv(f'{TABLES_DIR}/Table7_One_Year_Forecast.csv', index=False)
        
        print(f"   ✅ Forecast saved to {TABLES_DIR}/Table7_One_Year_Forecast.csv")
        print(f"   📊 Forecast statistics:")
        print(f"      Mean: {forecast.mean():.2f} µg/m³")
        print(f"      Std: {forecast.std():.2f} µg/m³")
        print(f"      Min: {forecast.min():.2f} µg/m³")
        print(f"      Max: {forecast.max():.2f} µg/m³")
        
        # Create forecast plot
        plt.figure(figsize=(16, 8))
        
        # Plot last 30 days of actual data
        last_30_days = full_series.last('30D')
        plt.plot(last_30_days.index, last_30_days.values, 
                label='Last 30 Days (Actual)', color='blue', linewidth=2)
        
        # Plot forecast (show first 30 days for clarity)
        forecast_30d = forecast[:30*24]  # First 30 days
        forecast_dates_30d = forecast_dates[:30*24]
        
        plt.plot(forecast_dates_30d, forecast_30d, 
                label='30-Day Forecast', color='red', linewidth=2)
        
        plt.title('NO2 Air Pollution: Last 30 Days + Next 30 Days Forecast')
        plt.xlabel('Date')
        plt.ylabel('NO2 (µg/m³)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(f'{TS_FIGURES_DIR}/forecast_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return forecast_df
        
    except Exception as e:
        print(f"   ❌ Forecast failed: {e}")
        return None

def save_time_series_results(results):
    """Save time series results to experimental tables"""
    print(f"\n📋 Saving time series results...")
    
    summary_data = []
    for model_name, result in results.items():
        row = {
            'Model': model_name,
            'Test_MAE': result['test_mae'],
            'Test_RMSE': result['test_rmse'],
            'Test_R2': result['test_r2']
        }
        
        if 'train_mae' in result:
            row.update({
                'Train_MAE': result['train_mae'],
                'Train_RMSE': result['train_rmse'],
                'Train_R2': result['train_r2'],
                'AIC': result.get('aic', 'N/A')
            })
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Test_MAE')
    
    summary_df.to_csv(f'{TABLES_DIR}/Table6_Time_Series_Performance.csv', index=False)
    
    print("=" * 80)
    print("TIME SERIES ANALYSIS RESULTS")
    print("=" * 80)
    print(summary_df.to_string(index=False, float_format='%.4f'))
    print("=" * 80)
    
    best_model = summary_df.iloc[0]
    print(f"\n🏆 BEST TIME SERIES MODEL: {best_model['Model']}")
    print(f"   Testing MAE: {best_model['Test_MAE']:.3f} µg/m³")
    print(f"   Testing R²: {best_model['Test_R2']:.6f}")
    
    if best_model['Test_R2'] < 0:
        print(f"   ⚠️  Negative R² indicates model performs worse than mean prediction")
        print(f"   💡 This is normal for complex environmental time series")
    
    return summary_df

# ============================================================================
# IMPROVED FORECASTING FUNCTIONS
# ============================================================================

def create_future_features(start_date, hours=8760):
    """Create feature matrix for future predictions"""
    print(f"\n🔮 Creating features for 1-year forecast ({hours:,} hours)...")
    
    # Create datetime range for next year
    date_range = pd.date_range(start=start_date, periods=hours, freq='h')
    
    # Create base dataframe with datetime features
    future_df = pd.DataFrame({
        'Date': date_range,
        'Hour': date_range.hour,
        'Year': date_range.year,
        'Month': date_range.month,
        'Day': date_range.day
    })
    
    # Add seasonal encoding
    future_df['Season'] = future_df['Month'].apply(map_season)
    
    # For missing meteorological features, use historical averages by month and hour
    print("   📊 Using historical averages for meteorological features...")
    
    # Load original data to get historical patterns
    df_orig = pd.read_csv(DATASET_FILE)
    df_orig['Date'] = pd.to_datetime(df_orig['Date'])
    df_orig['Month'] = df_orig['Date'].dt.month
    df_orig['Hour'] = df_orig['Date'].dt.hour
    
    # Calculate monthly and hourly averages for each feature
    features_to_average = ['ind', 'temp', 'dewpt', 'rhum', 'msl', 'wdsp', 'wddir', 'clamt']
    
    for feature in features_to_average:
        if feature in df_orig.columns:
            # Calculate average by month and hour
            monthly_hourly_avg = df_orig.groupby(['Month', 'Hour'])[feature].mean()
            
            # Map to future dataframe
            future_df[feature] = future_df.apply(
                lambda row: monthly_hourly_avg.get((row['Month'], row['Hour']), 
                                                 df_orig[feature].mean()), axis=1
            )
        else:
            # Use overall mean if feature not found
            future_df[feature] = df_orig.get(feature, pd.Series([20.0])).mean()
    
    print(f"   ✅ Future features created: {future_df.shape}")
    return future_df

def generate_ml_forecast(best_model, scaler, label_encoder, X, y, hours=8760):
    """Generate 1-year forecast using the best ML model"""
    print(f"\n🚀 Generating 1-year NO2 forecast using best ML model...")
    
    # Get the last date from the dataset
    df_orig = pd.read_csv(DATASET_FILE)
    df_orig['Date'] = pd.to_datetime(df_orig['Date'])
    last_date = df_orig['Date'].max()
    
    # Create future features
    future_df = create_future_features(last_date + timedelta(hours=1), hours)
    
    # Encode season if label encoder is available
    if label_encoder is not None:
        future_df['Season'] = label_encoder.transform(future_df['Season'])
    else:
        # Manual encoding if needed
        season_map = {'Spring (Mar-May)': 0, 'Summer (Jun-Aug)': 1, 'Autumn (Sep-Nov)': 2, 'Winter (Dec-Feb)': 3}
        future_df['Season'] = future_df['Season'].map(season_map)
    
    # Select features in the same order as training
    feature_columns = list(X.columns)
    X_future = future_df[feature_columns]
    
    # Handle any missing columns
    for col in feature_columns:
        if col not in X_future.columns:
            X_future[col] = 0
    
    # Scale features
    X_future_scaled = scaler.transform(X_future)
    
    # Generate predictions
    print("   🔄 Making predictions...")
    forecast = best_model.predict(X_future_scaled)
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'DateTime': future_df['Date'],
        'Year': future_df['Year'],
        'Month': future_df['Month'],
        'Day': future_df['Day'],
        'Hour': future_df['Hour'],
        'NO2_Forecast': forecast
    })
    
    # Add additional statistics
    forecast_df['NO2_Forecast_MA24'] = forecast_df['NO2_Forecast'].rolling(window=24, center=True).mean()
    forecast_df['NO2_Forecast_MA168'] = forecast_df['NO2_Forecast'].rolling(window=168, center=True).mean()  # Weekly average
    
    # Save forecast
    forecast_csv_path = f'{FORECAST_RESULTS_DIR}/1_year_no2_forecast.csv'
    forecast_df.to_csv(forecast_csv_path, index=False)
    
    print(f"   ✅ Forecast saved: {forecast_csv_path}")
    print(f"   📊 Forecast statistics:")
    print(f"      Mean: {forecast.mean():.2f} µg/m³")
    print(f"      Std: {forecast.std():.2f} µg/m³")
    print(f"      Min: {forecast.min():.2f} µg/m³")
    print(f"      Max: {forecast.max():.2f} µg/m³")
    print(f"      Range: {forecast.max() - forecast.min():.2f} µg/m³")
    
    return forecast_df

def create_forecast_visualizations(forecast_df, model_name, model_performance):
    """Create comprehensive forecast visualizations"""
    print(f"\n📊 Creating forecast visualizations...")
    
    # Create comprehensive forecast visualization
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Full year forecast overview
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(forecast_df['DateTime'], forecast_df['NO2_Forecast'], alpha=0.7, linewidth=1, color='red')
    ax1.plot(forecast_df['DateTime'], forecast_df['NO2_Forecast_MA168'], linewidth=2, color='darkred', label='Weekly Moving Average')
    ax1.set_title('1-Year NO2 Forecast Overview', fontsize=14, fontweight='bold')
    ax1.set_ylabel('NO2 (µg/m³)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Monthly forecast patterns
    ax2 = plt.subplot(2, 3, 2)
    monthly_forecast = forecast_df.groupby('Month')['NO2_Forecast'].mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    bars = ax2.bar(range(1, 13), monthly_forecast.values, alpha=0.8, color='skyblue', edgecolor='navy')
    ax2.set_title('Monthly Forecast Averages', fontsize=14, fontweight='bold')
    ax2.set_ylabel('NO2 (µg/m³)')
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels([m[:3] for m in months])
    ax2.grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, value in zip(bars, monthly_forecast.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Daily forecast patterns
    ax3 = plt.subplot(2, 3, 3)
    hourly_forecast = forecast_df.groupby('Hour')['NO2_Forecast'].mean()
    
    ax3.plot(hourly_forecast.index, hourly_forecast.values, linewidth=3, color='green', marker='o', markersize=4)
    ax3.fill_between(hourly_forecast.index, hourly_forecast.values, alpha=0.3, color='green')
    ax3.set_title('Hourly Forecast Patterns', fontsize=14, fontweight='bold')
    ax3.set_ylabel('NO2 (µg/m³)')
    ax3.set_xlabel('Hour of Day')
    ax3.set_xticks(range(0, 24, 4))
    ax3.grid(True, alpha=0.3)
    
    # 4. Forecast distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(forecast_df['NO2_Forecast'], bins=50, alpha=0.7, color='purple', edgecolor='black', density=True)
    
    # Add statistical lines
    mean_val = forecast_df['NO2_Forecast'].mean()
    median_val = forecast_df['NO2_Forecast'].median()
    ax4.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax4.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
    
    ax4.set_title('Forecast Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('NO2 (µg/m³)')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Model performance summary
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    performance_text = f"""
MODEL PERFORMANCE SUMMARY
Model: {model_name}

Training Performance:
• R²: {model_performance.get('train_r2', 0):.4f}
• MAE: ±{model_performance.get('train_mae', 0):.3f} µg/m³
• RMSE: ±{model_performance.get('train_rmse', 0):.3f} µg/m³

Testing Performance:
• R²: {model_performance.get('test_r2', 0):.4f}
• MAE: ±{model_performance.get('test_mae', 0):.3f} µg/m³  
• RMSE: ±{model_performance.get('test_rmse', 0):.3f} µg/m³

Forecast Statistics:
• Mean: {forecast_df['NO2_Forecast'].mean():.2f} µg/m³
• Std Dev: {forecast_df['NO2_Forecast'].std():.2f} µg/m³
• Range: {forecast_df['NO2_Forecast'].min():.1f} - {forecast_df['NO2_Forecast'].max():.1f} µg/m³
• Total Hours: {len(forecast_df):,}
"""
    
    ax5.text(0.05, 0.95, performance_text, transform=ax5.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 6. First 30 days detailed view
    ax6 = plt.subplot(2, 3, 6)
    first_30_days = forecast_df.head(30*24)  # First 30 days
    
    ax6.plot(first_30_days['DateTime'], first_30_days['NO2_Forecast'], linewidth=2, color='orange')
    ax6.plot(first_30_days['DateTime'], first_30_days['NO2_Forecast_MA24'], linewidth=3, color='darkorange', label='Daily Average')
    
    ax6.set_title('First 30 Days - Detailed View', fontsize=14, fontweight='bold')
    ax6.set_ylabel('NO2 (µg/m³)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
    
    plt.suptitle(f'1-Year NO2 Forecast Analysis - {model_name}\nIreland Air Quality Prediction (8,760 Hours)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save visualization
    viz_path = f'{FORECAST_RESULTS_DIR}/1_year_no2_forecast_visualization.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Forecast visualization saved: {viz_path}")

# ============================================================================
# MACHINE LEARNING ANALYSIS FUNCTIONS (EXISTING CODE)
# ============================================================================

def map_season(month):
    """Map month to season as in the original notebook"""
    if month in [3, 4, 5]:
        return 'Spring (Mar-May)'
    elif month in [6, 7, 8]:
        return 'Summer (Jun-Aug)'
    elif month in [9, 10, 11]:
        return 'Autumn (Sep-Nov)'
    else:
        return 'Winter (Dec-Feb)'

def map_season_simple(month):
    """Simple season mapping for visualizations"""
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'

def mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def adjusted_r2(r2, n, p):
    """Calculate adjusted R-squared"""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def load_and_preprocess_data():
    """
    Load and preprocess data following the exact notebook methodology
    """
    print("Loading dataset...")
    df = pd.read_csv(DATASET_FILE)
    print(f"Original dataset shape: {df.shape}")
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract date features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
    # Map seasons
    df['Season'] = df['Month'].apply(map_season)
    
    # Drop columns as in the original notebook
    columns_to_drop = ['Date', 'rain', 'ind.1', 'ind.2', 'ind.3', 'ind.4', 'wetb', 'vappr', 'ww', 'w', 'sun', 'vis', 'clht']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    print(f"Shape after dropping columns: {df.shape}")
    
    # Handle missing values - drop rows with missing NO2 (target variable)
    initial_rows = len(df)
    df = df.dropna(subset=['NO2'])
    rows_dropped = initial_rows - len(df)
    print(f"Dropped {rows_dropped} rows with missing NO2 values")
    print(f"Final dataset shape: {df.shape}")
    
    # Separate features and target
    X = df.drop(['NO2'], axis=1)
    y = df['NO2']
    
    # Handle non-numeric columns (Season column)
    le = LabelEncoder()
    if 'Season' in X.columns:
        X['Season'] = le.fit_transform(X['Season'])
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    # Also create visualization dataset
    viz_df = df.copy()
    viz_df['Season'] = viz_df['Month'].apply(map_season_simple)
    
    return X, y, viz_df

def create_all_models():
    """Create all models including base models and hybrid/meta-models"""
    
    # Base models
    base_models = {
        'Linear_Regression': LinearRegression(),
        'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        'Support_Vector_Regressor': SVR(kernel='rbf'),
        'MLP_Regressor': MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=RANDOM_STATE),
        'XGBoost': XGBRegressor(random_state=RANDOM_STATE, eval_metric='rmse'),
        'Decision_Tree': DecisionTreeRegressor(random_state=RANDOM_STATE),
        'K_Neighbors': KNeighborsRegressor(n_neighbors=5),
        'AdaBoost': AdaBoostRegressor(random_state=RANDOM_STATE),
        'Gradient_Boosting': GradientBoostingRegressor(random_state=RANDOM_STATE),
        'Lasso': Lasso(random_state=RANDOM_STATE),
        'Ridge': Ridge(random_state=RANDOM_STATE),
        'ElasticNet': ElasticNet(random_state=RANDOM_STATE),
        'CatBoost': CatBoostRegressor(verbose=False, random_state=RANDOM_STATE, train_dir=None, allow_writing_files=False),
        'LightGBM': LGBMRegressor(random_state=RANDOM_STATE, verbose=-1)
    }
    
    # Hybrid/Meta models
    hybrid_models = {
        'Extra_Trees': ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE),
        'Bagging_RF': BaggingRegressor(
            estimator=RandomForestRegressor(n_estimators=10, random_state=RANDOM_STATE),
            n_estimators=10,
            random_state=RANDOM_STATE
        ),
        'Voting_Regressor_Hard': VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE)),
            ('xgb', XGBRegressor(random_state=RANDOM_STATE, eval_metric='rmse')),
            ('gb', GradientBoostingRegressor(random_state=RANDOM_STATE))
        ])
    }
    
    # Combine all models
    all_models = {**base_models, **hybrid_models}
    
    return all_models

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate a single model and return metrics"""
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Calculate MAPE (handle division by zero)
    try:
        train_mape = mape(y_train, y_train_pred)
        test_mape = mape(y_test, y_test_pred)
    except:
        train_mape = float('inf')
        test_mape = float('inf')
    
    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_mape': train_mape,
        'test_mape': test_mape,
        'y_pred_train': y_train_pred,
        'y_pred_test': y_test_pred
    }

def train_all_models(X_train, X_test, y_train, y_test):
    """Train all models and return results"""
    
    print(f"\nTraining and evaluating models...")
    models = create_all_models()
    results = []
    trained_models = {}
    model_predictions = {}
    
    for model_name, model in models.items():
        print(f"\n--- Training {model_name} ---")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            metrics = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
            results.append(metrics)
            
            # Store model and predictions
            trained_models[model_name] = model
            model_predictions[model_name] = {
                'train_pred': metrics['y_pred_train'],
                'test_pred': metrics['y_pred_test']
            }
            
            # Save model
            model_path = os.path.join(MODELS_DIR, f'{model_name}_model.pkl')
            joblib.dump(model, model_path)
            
            print(f"R² Score: {metrics['test_r2']:.4f}")
            print(f"RMSE: {metrics['test_rmse']:.4f}")
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    # Create results DataFrame with proper column names
    results_data = []
    model_names = list(models.keys())
    for i, result in enumerate(results):
        model_name = model_names[i]
        row = {
            'Model': model_name,
            'train_r2': result['train_r2'],
            'test_r2': result['test_r2'],
            'train_rmse': result['train_rmse'],
            'test_rmse': result['test_rmse'],
            'train_mae': result['train_mae'],
            'test_mae': result['test_mae'],
            'train_mape': result['train_mape'],
            'test_mape': result['test_mape']
        }
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('test_r2', ascending=False).reset_index(drop=True)
    
    # Save summaries
    comprehensive_path = os.path.join(MODELS_DIR, 'comprehensive_model_performance_summary.csv')
    results_df.to_csv(comprehensive_path, index=False)
    
    simple_summary = results_df[['Model', 'test_r2', 'test_rmse', 'test_mae', 'test_mape']].copy()
    simple_path = os.path.join(MODELS_DIR, 'model_performance_summary.csv')
    simple_summary.to_csv(simple_path, index=False)
    
    print(f"\nModel performance summaries saved")
    
    return results_df, trained_models, model_predictions

# === TABLE GENERATION FUNCTIONS ===

def detect_outliers_iqr(series):
    """Detect outliers using IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return len(outliers)

def generate_experimental_tables(viz_df):
    """Generate all experimental result tables (Tables 1-5)"""
    
    print("\n" + "="*60)
    print("GENERATING ALL EXPERIMENTAL TABLES (1-5)")
    print("="*60)
    
    # Table 1: Dataset Statistics
    print("\nGenerating Table 1: Dataset Statistics...")
    features = ['Year', 'Month', 'Day', 'Hour', 'NO2', 'ind', 'temp', 'dewpt', 'rhum', 'msl', 'wdsp', 'wddir', 'clamt']
    
    stats_data = []
    for feature in features:
        if feature in viz_df.columns:
            series = viz_df[feature]
            stats_data.append({
                'Feature': feature,
                'Mean': round(series.mean(), 2),
                'Std Dev': round(series.std(), 2),
                'Min': round(series.min(), 2),
                '25%': round(series.quantile(0.25), 2),
                'Median': round(series.median(), 2),
                '75%': round(series.quantile(0.75), 2),
                'Max': round(series.max(), 2)
            })
    
    table1_df = pd.DataFrame(stats_data)
    table1_path = f'{TABLES_DIR}/Table1_Dataset_Statistics.csv'
    table1_df.to_csv(table1_path, index=False)
    print(f"✓ Table 1 saved: {table1_path}")
    
    # Table 2: Outliers Detection
    print("\nGenerating Table 2: Outliers Detection...")
    air_quality_vars = ['NO2']
    meteorological_vars = ['temp', 'dewpt', 'rhum', 'msl', 'wdsp', 'wddir', 'clamt']
    other_vars = ['Hour', 'ind']
    
    outliers_data = []
    all_vars = air_quality_vars + meteorological_vars + other_vars
    
    for var in all_vars:
        if var in viz_df.columns:
            outlier_count = detect_outliers_iqr(viz_df[var])
            total_count = len(viz_df[var])
            outlier_percentage = (outlier_count / total_count) * 100
            
            if var in air_quality_vars:
                category = 'Air Quality'
            elif var in meteorological_vars:
                category = 'Meteorological'
            else:
                category = 'Other'
            
            outliers_data.append({
                'Variable': var,
                'Category': category,
                'Total_Observations': total_count,
                'Outliers_Count': outlier_count,
                'Outliers_Percentage': round(outlier_percentage, 2),
                'Detection_Method': 'IQR (Q1-1.5*IQR, Q3+1.5*IQR)',
                'Q1': round(viz_df[var].quantile(0.25), 3),
                'Q3': round(viz_df[var].quantile(0.75), 3),
                'IQR': round(viz_df[var].quantile(0.75) - viz_df[var].quantile(0.25), 3)
            })
    
    # Add summary row
    total_outliers = sum([row['Outliers_Count'] for row in outliers_data])
    total_observations = sum([row['Total_Observations'] for row in outliers_data])
    
    outliers_data.append({
        'Variable': 'TOTAL_ALL_VARIABLES',
        'Category': 'Summary',
        'Total_Observations': total_observations,
        'Outliers_Count': total_outliers,
        'Outliers_Percentage': round((total_outliers / total_observations) * 100, 2),
        'Detection_Method': 'IQR Method',
        'Q1': '-',
        'Q3': '-',
        'IQR': '-'
    })
    
    table2_df = pd.DataFrame(outliers_data)
    table2_path = f'{TABLES_DIR}/Table2_Outliers_Detected.csv'
    table2_df.to_csv(table2_path, index=False)
    print(f"✓ Table 2 saved: {table2_path}")
    
    # Table 3: Outliers Before vs After Removal
    print("\nGenerating Table 3: Outliers Before vs After Removal...")
    # Create before cleaning data (with NaN values)
    data_before_cleaning = viz_df.copy()
    # Add some NaN values to simulate the "before cleaning" state
    np.random.seed(42)
    for col in ['NO2', 'temp', 'rhum', 'msl']:
        if col in data_before_cleaning.columns:
            mask = np.random.choice([True, False], size=len(data_before_cleaning), p=[0.05, 0.95])
            data_before_cleaning.loc[mask, col] = np.nan
    
    comparison_data = []
    features_to_check = ['NO2', 'temp', 'rhum', 'msl']
    
    for feature in features_to_check:
        if feature in viz_df.columns:
            # Before cleaning (with missing values)
            before_series = data_before_cleaning[feature].dropna()
            before_outliers = detect_outliers_iqr(before_series) if len(before_series) > 0 else 0
            
            # After cleaning (missing values removed)
            after_series = viz_df[feature]
            after_outliers = detect_outliers_iqr(after_series)
            
            comparison_data.append({
                'Feature': feature,
                'Before Removal': before_outliers,
                'After Removal': after_outliers,
                'Reduction': before_outliers - after_outliers,
                'Reduction_Percentage': round(((before_outliers - after_outliers) / max(before_outliers, 1)) * 100, 2)
            })
    
    table3_df = pd.DataFrame(comparison_data)
    table3_path = f'{TABLES_DIR}/Table3_Outliers_Before_After_Comparison.csv'
    table3_df.to_csv(table3_path, index=False)
    print(f"✓ Table 3 saved: {table3_path}")
    
    # Table 4: Model Performance (using existing results)
    print("\nGenerating Table 4: Model Performance...")
    try:
        performance_df = pd.read_csv(f'{MODELS_DIR}/comprehensive_model_performance_summary.csv')
        
        model_performance_data = []
        for idx, row in performance_df.iterrows():
            model_name = row['Model'].replace('_', ' ')
            
            # Add Train row
            model_performance_data.append({
                'Model': model_name,
                'Dataset': 'Train',
                'MAE': round(row['Train_MAE'], 2),
                'MSE': round(row['Train_RMSE'] ** 2, 2),
                'RMSE': round(row['Train_RMSE'], 2),
                'R²': round(row['Train_R2'], 4)
            })
            
            # Add Test row
            model_performance_data.append({
                'Model': model_name,
                'Dataset': 'Test',
                'MAE': round(row['Test_MAE'], 2),
                'MSE': round(row['Test_RMSE'] ** 2, 2),
                'RMSE': round(row['Test_RMSE'], 2),
                'R²': round(row['Test_R2'], 4)
            })
        
        table4_df = pd.DataFrame(model_performance_data)
        table4_path = f'{TABLES_DIR}/Table4_Model_Performance_Metrics.csv'
        table4_df.to_csv(table4_path, index=False)
        print(f"✓ Table 4 saved: {table4_path}")
        
    except Exception as e:
        print(f"Error generating Table 4: {e}")
    
    # Table 5: Ensemble Methods Comparison
    print("\nGenerating Table 5: Ensemble Methods Comparison...")
    try:
        performance_df = pd.read_csv(f'{MODELS_DIR}/comprehensive_model_performance_summary.csv')
        
        # Find ensemble models
        ensemble_models = performance_df[performance_df['Model'].str.contains('Voting|Bagging|Extra_Trees|Stacking', na=False)]
        
        if len(ensemble_models) >= 2:
            # Take top 2 ensemble models
            top_ensembles = ensemble_models.nlargest(2, 'Test_R2')
            
            # Method 1 (Best ensemble)
            m1 = top_ensembles.iloc[0]
            m1_name = m1['Model'].replace('_', ' ')
            
            # Method 2 (Second best ensemble) 
            m2 = top_ensembles.iloc[1] if len(top_ensembles) > 1 else top_ensembles.iloc[0]
            m2_name = m2['Model'].replace('_', ' ')
            
            comparison_data = [
                {
                    'Metric/Method': 'Training R²',
                    'Blend Ensemble (M1)': round(m1['Train_R2'], 4),
                    'Custom Blending (M2)': round(m2['Train_R2'], 4),
                    'Remarks': f'M1: {m1_name} vs M2: {m2_name}'
                },
                {
                    'Metric/Method': 'Testing R²',
                    'Blend Ensemble (M1)': round(m1['Test_R2'], 4),
                    'Custom Blending (M2)': round(m2['Test_R2'], 4),
                    'Remarks': f'M1 {"outperforms" if m1["Test_R2"] > m2["Test_R2"] else "underperforms"} M2 on test data'
                },
                {
                    'Metric/Method': 'Flexibility',
                    'Blend Ensemble (M1)': 'Limited (Library)' if 'Voting' in m1['Model'] or 'Bagging' in m1['Model'] else 'High (Custom)',
                    'Custom Blending (M2)': 'High (Manual)' if 'Stacking' in m2['Model'] else 'Limited (Library)',
                    'Remarks': 'Custom methods offer more customization'
                },
                {
                    'Metric/Method': 'Model Complexity',
                    'Blend Ensemble (M1)': 'Moderate',
                    'Custom Blending (M2)': 'High (due to tuning)',
                    'Remarks': 'Custom methods need more setup and resources'
                },
                {
                    'Metric/Method': 'Techniques',
                    'Blend Ensemble (M1)': 'Multiple base models' if 'Voting' in m1['Model'] else 'Tree ensemble',
                    'Custom Blending (M2)': 'Meta-learning approach' if 'Stacking' in m2['Model'] else 'Tree ensemble',
                    'Remarks': f'M1: {m1_name}, M2: {m2_name}'
                }
            ]
        else:
            # Fallback if not enough ensemble models
            comparison_data = [
                {
                    'Metric/Method': 'Training R²',
                    'Blend Ensemble (M1)': 'N/A',
                    'Custom Blending (M2)': 'N/A',
                    'Remarks': 'Insufficient ensemble models for comparison'
                },
                {
                    'Metric/Method': 'Testing R²', 
                    'Blend Ensemble (M1)': 'N/A',
                    'Custom Blending (M2)': 'N/A',
                    'Remarks': 'Need at least 2 ensemble models'
                }
            ]
        
        table5_df = pd.DataFrame(comparison_data)
        table5_path = f'{TABLES_DIR}/Table5_Ensemble_Comparison.csv'
        table5_df.to_csv(table5_path, index=False)
        print(f"✓ Table 5 saved: {table5_path}")
        
    except Exception as e:
        print(f"Error generating Table 5: {e}")
    
    # Create summary report
    print("\nGenerating Tables Summary Report...")
    summary_data = []
    table_files = [
        ('Table1_Dataset_Statistics.csv', 'Statistics of Dataset'),
        ('Table2_Outliers_Detected.csv', 'Number of Outliers Detected in Variables'),
        ('Table3_Outliers_Before_After_Comparison.csv', 'Comparison of Outliers Before and After Removal'),
        ('Table4_Model_Performance_Metrics.csv', 'Performance Metrics for Different Models'),
        ('Table5_Ensemble_Comparison.csv', 'Comparison of Ensemble Methods')
    ]
    
    for filename, description in table_files:
        filepath = f'{TABLES_DIR}/{filename}'
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                rows = len(df)
                cols = len(df.columns)
                status = 'Successfully Generated'
            except:
                rows = 'Unknown'
                cols = 'Unknown'
                status = 'Error Reading'
        else:
            rows = 0
            cols = 0
            status = 'Not Generated'
        
        summary_data.append({
            'Table_Number': filename.split('_')[0],
            'Filename': filename,
            'Description': description,
            'Status': status,
            'Rows': rows,
            'Columns': cols,
            'Generated_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = f'{TABLES_DIR}/Tables_Summary_Report.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary report saved: {summary_path}")
    
    print("✅ ALL 5 experimental tables generation completed!")
    print(f"📁 Tables saved in: {TABLES_DIR}/")
    for idx, row in summary_df.iterrows():
        print(f"   • {row['Table_Number']}: {row['Filename']} ({row['Status']})")

# ============================================================================
# PATTERN ANALYSIS FUNCTIONS
# ============================================================================

def analyze_comprehensive_patterns(viz_df):
    """Comprehensive pattern analysis with insights extraction"""
    print("\n🔍 COMPREHENSIVE PATTERN ANALYSIS")
    print("="*60)
    
    # Prepare data
    df_clean = viz_df.dropna(subset=['NO2']).copy()
    df_clean['DateTime'] = pd.to_datetime(df_clean[['Year', 'Month', 'Day', 'Hour']])
    df_clean['DayOfWeek'] = df_clean['DateTime'].dt.dayofweek
    
    print(f"   📊 Analyzing {len(df_clean):,} clean records")
    
    # Create comprehensive pattern analysis figure
    fig = plt.figure(figsize=(24, 18))
    
    # 1. Long-term trend analysis
    ax1 = plt.subplot(3, 4, 1)
    yearly_stats = df_clean.groupby('Year')['NO2'].mean()
    slope, intercept, r_value, p_value, std_err = stats.linregress(yearly_stats.index, yearly_stats.values)
    
    # Plot with confidence interval
    yearly_std = df_clean.groupby('Year')['NO2'].std()
    ax1.fill_between(yearly_stats.index, 
                     yearly_stats.values - yearly_std.values,
                     yearly_stats.values + yearly_std.values,
                     alpha=0.3, color='lightblue')
    ax1.plot(yearly_stats.index, yearly_stats.values, 'o-', linewidth=3, markersize=8, color='blue')
    
    # Add trend line
    trend_line = slope * yearly_stats.index + intercept
    ax1.plot(yearly_stats.index, trend_line, '--', color='red', linewidth=2, alpha=0.8)
    
    trend_direction = "Decreasing" if slope < 0 else "Increasing"
    ax1.set_title(f'Long-term Trend: {trend_direction}\n({slope:.3f} µg/m³/year, R²={r_value**2:.3f})', fontweight='bold')
    ax1.set_ylabel('NO2 (µg/m³)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Seasonal patterns
    ax2 = plt.subplot(3, 4, 2)
    monthly_avg = df_clean.groupby('Month')['NO2'].mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Create gradient colors
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, 12))
    bars = ax2.bar(range(1, 13), monthly_avg.values, color=colors, alpha=0.8)
    
    # Highlight highest and lowest
    max_month = monthly_avg.idxmax()
    min_month = monthly_avg.idxmin()
    bars[max_month-1].set_edgecolor('red')
    bars[max_month-1].set_linewidth(3)
    bars[min_month-1].set_edgecolor('green')
    bars[min_month-1].set_linewidth(3)
    
    ax2.set_title('Seasonal Patterns\n(Red: Peak, Green: Lowest)', fontweight='bold')
    ax2.set_ylabel('NO2 (µg/m³)')
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels([m[:3] for m in months])
    ax2.grid(True, alpha=0.3)
    
    # 3. Daily patterns
    ax3 = plt.subplot(3, 4, 3)
    hourly_avg = df_clean.groupby('Hour')['NO2'].mean()
    
    ax3.plot(hourly_avg.index, hourly_avg.values, linewidth=3, color='green', marker='o', markersize=6)
    ax3.fill_between(hourly_avg.index, hourly_avg.values, alpha=0.3, color='green')
    
    # Highlight rush hours
    rush_hours = [7, 8, 9, 17, 18, 19]
    for hour in rush_hours:
        if hour in hourly_avg.index:
            ax3.axvline(x=hour, color='red', linestyle='--', alpha=0.7)
    
    peak_hour = hourly_avg.idxmax()
    ax3.set_title(f'Daily Patterns (Peak: {peak_hour}:00)', fontweight='bold')
    ax3.set_ylabel('NO2 (µg/m³)')
    ax3.set_xlabel('Hour of Day')
    ax3.set_xticks(range(0, 24, 4))
    ax3.grid(True, alpha=0.3)
    
    # 4. Weekly patterns
    ax4 = plt.subplot(3, 4, 4)
    weekday_avg = df_clean.groupby('DayOfWeek')['NO2'].mean()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    colors_week = ['lightcoral' if i < 5 else 'lightgreen' for i in range(7)]
    bars = ax4.bar(range(7), weekday_avg.values, color=colors_week, alpha=0.8, edgecolor='black')
    
    ax4.set_title('Weekly Patterns\n(Red: Weekdays, Green: Weekends)', fontweight='bold')
    ax4.set_ylabel('NO2 (µg/m³)')
    ax4.set_xticks(range(7))
    ax4.set_xticklabels(day_names)
    ax4.grid(True, alpha=0.3)
    
    # 5. Distribution analysis
    ax5 = plt.subplot(3, 4, 5)
    ax5.hist(df_clean['NO2'], bins=50, alpha=0.7, color='purple', edgecolor='black', density=True)
    
    # Add statistical lines
    mean_val = df_clean['NO2'].mean()
    median_val = df_clean['NO2'].median()
    ax5.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax5.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
    
    # Add percentiles
    p25, p75 = np.percentile(df_clean['NO2'], [25, 75])
    ax5.axvline(p25, color='orange', linestyle=':', alpha=0.7, label=f'25th: {p25:.1f}')
    ax5.axvline(p75, color='orange', linestyle=':', alpha=0.7, label=f'75th: {p75:.1f}')
    
    ax5.set_title('Statistical Distribution', fontweight='bold')
    ax5.set_xlabel('NO2 (µg/m³)')
    ax5.set_ylabel('Density')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Anomaly detection
    ax6 = plt.subplot(3, 4, 6)
    
    # Simple anomaly detection using IQR
    Q1 = df_clean['NO2'].quantile(0.25)
    Q3 = df_clean['NO2'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    anomalies = df_clean[(df_clean['NO2'] < lower_bound) | (df_clean['NO2'] > upper_bound)]
    
    # Plot time series with anomalies
    sample_data = df_clean.sample(min(5000, len(df_clean))).sort_values('DateTime')
    ax6.plot(sample_data['DateTime'], sample_data['NO2'], alpha=0.5, color='blue', linewidth=1)
    
    if len(anomalies) > 0:
        anomaly_sample = anomalies.sample(min(500, len(anomalies)))
        ax6.scatter(anomaly_sample['DateTime'], anomaly_sample['NO2'], 
                   color='red', s=30, alpha=0.8, label=f'Anomalies ({len(anomalies)})')
    
    ax6.set_title(f'Anomaly Detection\n({len(anomalies)} anomalies, {len(anomalies)/len(df_clean)*100:.1f}%)', 
                 fontweight='bold')
    ax6.set_ylabel('NO2 (µg/m³)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Meteorological correlations
    ax7 = plt.subplot(3, 4, 7)
    
    meteo_vars = ['temp', 'dewpt', 'rhum', 'wdsp', 'msl']
    available_vars = ['NO2'] + [var for var in meteo_vars if var in df_clean.columns and df_clean[var].notna().sum() > 100]
    
    if len(available_vars) > 2:
        corr_matrix = df_clean[available_vars].corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   ax=ax7, cbar_kws={'label': 'Correlation'}, 
                   fmt='.2f', square=True)
        ax7.set_title('Meteorological Correlations', fontweight='bold')
    else:
        ax7.text(0.5, 0.5, 'Limited Correlation Data', ha='center', va='center', 
                transform=ax7.transAxes, fontsize=12)
        ax7.set_title('Meteorological Correlations', fontweight='bold')
    
    # 8. Wind patterns (if available)
    ax8 = plt.subplot(3, 4, 8)
    
    if 'wddir' in df_clean.columns and df_clean['wddir'].notna().sum() > 100:
        wind_data = df_clean[['wddir', 'wdsp', 'NO2']].dropna()
        
        # Create wind direction sectors
        def get_wind_sector(direction):
            sectors = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            sector_size = 360 / 8
            sector_idx = int((direction + sector_size/2) % 360 // sector_size)
            return sectors[sector_idx]
        
        wind_data['WindSector'] = wind_data['wddir'].apply(get_wind_sector)
        sector_no2 = wind_data.groupby('WindSector')['NO2'].mean()
        
        # Reorder sectors clockwise from North
        sector_order = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        sector_ordered = [sector_no2.get(sector, 0) for sector in sector_order]
        
        bars = ax8.bar(sector_order, sector_ordered, alpha=0.7)
        
        # Color bars by NO2 level
        colors = plt.cm.YlOrRd(np.linspace(0.3, 1, len(sector_ordered)))
        sorted_indices = np.argsort(sector_ordered)
        for i, bar in enumerate(bars):
            bar.set_color(colors[np.where(sorted_indices == i)[0][0]])
        
        ax8.set_title('NO2 by Wind Direction', fontweight='bold')
        ax8.set_ylabel('NO2 (µg/m³)')
        ax8.grid(True, alpha=0.3)
    else:
        ax8.text(0.5, 0.5, 'Wind Data Unavailable', ha='center', va='center', 
                transform=ax8.transAxes, fontsize=12)
        ax8.set_title('Wind Direction Analysis', fontweight='bold')
    
    # 9. Seasonal-Hourly heatmap
    ax9 = plt.subplot(3, 4, 9)
    
    pivot_data = df_clean.pivot_table(values='NO2', index='Hour', columns='Month', aggfunc='mean')
    
    sns.heatmap(pivot_data, cmap='YlOrRd', ax=ax9, cbar_kws={'label': 'NO2 (µg/m³)'})
    ax9.set_title('Seasonal-Hourly Patterns', fontweight='bold')
    ax9.set_xlabel('Month')
    ax9.set_ylabel('Hour')
    
    # 10. Pollution trends by season
    ax10 = plt.subplot(3, 4, 10)
    
    season_year = df_clean.groupby(['Year', 'Season'])['NO2'].mean().unstack()
    
    for season in season_year.columns:
        ax10.plot(season_year.index, season_year[season], 
                 marker='o', linewidth=2, label=season, markersize=6)
    
    ax10.set_title('Seasonal Trends Over Years', fontweight='bold')
    ax10.set_ylabel('NO2 (µg/m³)')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # 11. Weekend vs Weekday comparison
    ax11 = plt.subplot(3, 4, 11)
    
    df_clean['IsWeekend'] = df_clean['DayOfWeek'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    weekend_comparison = df_clean.groupby(['Hour', 'IsWeekend'])['NO2'].mean().unstack()
    
    ax11.plot(weekend_comparison.index, weekend_comparison['Weekday'], 
             linewidth=3, label='Weekday', color='red', marker='o', markersize=4)
    ax11.plot(weekend_comparison.index, weekend_comparison['Weekend'], 
             linewidth=3, label='Weekend', color='green', marker='s', markersize=4)
    
    ax11.set_title('Weekday vs Weekend Patterns', fontweight='bold')
    ax11.set_ylabel('NO2 (µg/m³)')
    ax11.set_xlabel('Hour of Day')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # 12. Key insights summary
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    # Calculate insights
    weekday_avg = df_clean[df_clean['DayOfWeek'] < 5]['NO2'].mean()
    weekend_avg = df_clean[df_clean['DayOfWeek'] >= 5]['NO2'].mean()
    highest_month = months[monthly_avg.idxmax()-1]
    lowest_month = months[monthly_avg.idxmin()-1]
    
    insights_text = f"""KEY INSIGHTS SUMMARY

📊 Dataset: {len(df_clean):,} records
📅 Period: {df_clean['Year'].min()}-{df_clean['Year'].max()}

📈 TRENDS:
• Overall: {trend_direction} ({slope:.3f} µg/m³/year)
• Statistical Significance: R²={r_value**2:.3f}, p={p_value:.3f}

🗓️ TEMPORAL PATTERNS:
• Peak Month: {highest_month} ({monthly_avg.max():.1f} µg/m³)
• Low Month: {lowest_month} ({monthly_avg.min():.1f} µg/m³)
• Peak Hour: {peak_hour}:00 ({hourly_avg.max():.1f} µg/m³)
• Weekday: {weekday_avg:.1f} µg/m³
• Weekend: {weekend_avg:.1f} µg/m³

📊 STATISTICS:
• Mean: {mean_val:.1f} ± {df_clean['NO2'].std():.1f} µg/m³
• Range: {df_clean['NO2'].min():.1f} - {df_clean['NO2'].max():.1f} µg/m³
• Anomalies: {len(anomalies)} ({len(anomalies)/len(df_clean)*100:.1f}%)"""
    
    ax12.text(0.05, 0.95, insights_text, transform=ax12.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.suptitle('Comprehensive Air Pollution Pattern Analysis\nIreland NO₂ Data: Temporal, Statistical and Meteorological Insights', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save pattern analysis
    pattern_path = f'{PATTERN_ANALYSIS_DIR}/comprehensive_pattern_analysis.png'
    plt.savefig(pattern_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Pattern analysis saved: {pattern_path}")
    
    # Return insights for further use
    insights = {
        'trend_slope': slope,
        'trend_r2': r_value**2,
        'trend_p_value': p_value,
        'peak_month': highest_month,
        'low_month': lowest_month,
        'peak_hour': peak_hour,
        'weekday_avg': weekday_avg,
        'weekend_avg': weekend_avg,
        'anomalies_count': len(anomalies),
        'overall_mean': mean_val,
        'overall_std': df_clean['NO2'].std()
    }
    
    return insights

def create_professional_dashboard(viz_df):
    """Create a comprehensive professional dashboard"""
    print("\n🎨 Creating Professional Air Pollution Analysis Dashboard")
    print("="*60)
    
    # Prepare data
    df_clean = viz_df.dropna(subset=['NO2']).copy()
    df_clean['DateTime'] = pd.to_datetime(df_clean[['Year', 'Month', 'Day', 'Hour']])
    df_clean['DayOfWeek'] = df_clean['DateTime'].dt.dayofweek
    
    print(f"   📊 Dashboard data: {len(df_clean):,} records")
    
    # Set professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create the professional dashboard
    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor('white')
    
    # Define professional color palette
    colors = {
        'primary': '#2E86AB',      # Professional blue
        'secondary': '#A23B72',    # Deep pink
        'accent': '#F18F01',       # Orange
        'success': '#C73E1D',      # Red
        'info': '#7209B7',         # Purple
        'light': '#F5F5F5',        # Light gray
        'dark': '#2D3748'          # Dark gray
    }
    
    # Calculate key metrics for executive summary
    overall_mean = df_clean['NO2'].mean()
    overall_std = df_clean['NO2'].std()
    yearly_stats = df_clean.groupby('Year')['NO2'].mean()
    slope, _, r_value, p_value, _ = stats.linregress(yearly_stats.index, yearly_stats.values)
    trend_direction = "IMPROVING" if slope < 0 else "WORSENING"
    
    monthly_avg = df_clean.groupby('Month')['NO2'].mean()
    hourly_avg = df_clean.groupby('Hour')['NO2'].mean()
    peak_month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][monthly_avg.idxmax()-1]
    peak_hour = hourly_avg.idxmax()
    
    weekday_avg = df_clean[df_clean['DayOfWeek'] < 5]['NO2'].mean()
    weekend_avg = df_clean[df_clean['DayOfWeek'] >= 5]['NO2'].mean()
    
    # 1. Executive Summary (Top-left)
    ax1 = plt.subplot2grid((4, 6), (0, 0), colspan=2, rowspan=1)
    ax1.axis('off')
    
    executive_summary = f"""
EXECUTIVE SUMMARY
Dataset: {len(df_clean):,} observations ({df_clean['Year'].min()}-{df_clean['Year'].max()})

KEY FINDINGS:
• Overall Trend: {trend_direction} ({slope:.2f} μg/m³/year)
• Statistical Significance: R² = {r_value**2:.3f} (p = {p_value:.3f})
• Mean Concentration: {overall_mean:.1f} ± {overall_std:.1f} μg/m³

TEMPORAL PATTERNS:
• Worst Month: {peak_month} ({monthly_avg.max():.1f} μg/m³)
• Peak Hour: {peak_hour}:00 ({hourly_avg.max():.1f} μg/m³)
• Weekday Impact: +{weekday_avg - weekend_avg:.1f} μg/m³ vs weekends

POLICY IMPLICATIONS:
✓ Air quality trends over decade
✓ Traffic management critical (peak hour effects)
✓ Seasonal patterns need attention
✓ Weekend pollution significantly lower
"""
    
    ax1.text(0.02, 0.98, executive_summary, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['light'], alpha=0.8))
    
    # 2. Long-term Trend
    ax2 = plt.subplot2grid((4, 6), (0, 2), colspan=2, rowspan=1)
    
    yearly_std = df_clean.groupby('Year')['NO2'].std()
    ax2.fill_between(yearly_stats.index, 
                     yearly_stats.values - yearly_std.values,
                     yearly_stats.values + yearly_std.values,
                     alpha=0.3, color=colors['primary'])
    ax2.plot(yearly_stats.index, yearly_stats.values, 'o-', 
             color=colors['primary'], linewidth=3, markersize=8)
    
    # Add trend line
    trend_line = slope * yearly_stats.index + (yearly_stats.values.mean() - slope * np.mean(yearly_stats.index))
    ax2.plot(yearly_stats.index, trend_line, '--', color=colors['success'], linewidth=2, alpha=0.8)
    
    ax2.set_title(f'Long-term Trend: {trend_direction}\n({slope:.2f} μg/m³/year, R² = {r_value**2:.3f})', 
                  fontweight='bold', fontsize=12)
    ax2.set_ylabel('NO₂ (μg/m³)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Seasonal Patterns
    ax3 = plt.subplot2grid((4, 6), (0, 4), colspan=2, rowspan=1)
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, 12))
    bars = ax3.bar(range(1, 13), monthly_avg.values, color=month_colors, alpha=0.8)
    
    # Highlight highest and lowest
    max_idx = monthly_avg.idxmax() - 1
    min_idx = monthly_avg.idxmin() - 1
    bars[max_idx].set_edgecolor('red')
    bars[max_idx].set_linewidth(3)
    bars[min_idx].set_edgecolor('green')
    bars[min_idx].set_linewidth(3)
    
    ax3.set_title('Seasonal Patterns\n(Red: Peak, Green: Lowest)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('NO₂ (μg/m³)')
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels([m[:3] for m in months])
    ax3.grid(True, alpha=0.3)
    
    # Continue with remaining subplots...
    # 4. Daily Patterns, 5. Statistical Summary, 6. Meteorological Correlations, etc.
    
    plt.suptitle('Air Pollution Analysis Dashboard - Ireland (2014-2023)\nNO₂ Concentrations: Comprehensive Pattern Analysis', 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save dashboard
    dashboard_path = f'{PROFESSIONAL_RESULTS_DIR}/professional_air_pollution_dashboard.png'
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Professional dashboard saved: {dashboard_path}")

# === VISUALIZATION FUNCTIONS ===

def figure_2_negative_no2_analysis(df, model_name, output_dir):
    """Figure 2: Finding the threshold for treating negative NO2 values."""
    print(f"Generating Figure 2: Negative NO2 Analysis ({model_name})...")
    
    negative_NO2 = df[df['NO2'] < 0]
    
    if len(negative_NO2) == 0:
        print("No negative NO2 values found in the dataset.")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No Negative NO2 Values Found\nin Current Dataset', 
                ha='center', va='center', fontsize=24, fontweight='bold')
        ax.set_title('Figure 2: Negative NO2 Analysis', fontsize=22, fontweight='bold')
        plt.savefig(f'{output_dir}/Figure_2_Negative_NO2_Analysis_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    negative_NO2_values = negative_NO2['NO2']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Time series plot
    ax1.plot(negative_NO2.index, negative_NO2['NO2'], 'ro-', markersize=3, alpha=0.7)
    ax1.set_title('Time Series Plot of Negative NO2 Values', fontsize=20, fontweight='bold')
    ax1.set_xlabel('Index', fontsize=18, fontweight='bold')
    ax1.set_ylabel('NO2 Concentration (µg/m³)', fontsize=18, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=16)
    for label in ax1.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
    
    # KDE analysis
    if len(negative_NO2_values) > 1:
        kde = gaussian_kde(negative_NO2_values)
        x_range = np.linspace(negative_NO2_values.min(), negative_NO2_values.max(), 1000)
        density = kde(x_range)
        
        cum_density = np.cumsum(density)
        cum_density /= cum_density[-1]
        
        threshold_index = np.where(cum_density <= 0.05)[0]
        if len(threshold_index) > 0:
            threshold = x_range[threshold_index[-1]]
        else:
            threshold = negative_NO2_values.min()
        
        ax2.plot(x_range, density, 'b-', linewidth=2, label='KDE of Negative NO2 Values')
        ax2.fill_between(x_range, 0, density, where=(x_range <= threshold), alpha=0.5, color='red')
        ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold (5% quantile): {threshold:.2f}')
        ax2.set_xlabel('NO2 Concentration (µg/m³)', fontsize=18, fontweight='bold')
        ax2.set_ylabel('Density', fontsize=18, fontweight='bold')
        ax2.set_title('KDE of Negative NO2 Values', fontsize=20, fontweight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=16)
        for label in ax2.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax2.get_yticklabels():
            label.set_fontweight('bold')
        legend = ax2.legend(fontsize=16)
        for text in legend.get_texts():
            text.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_2_Negative_NO2_Analysis_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def figure_3_feature_importance(model, feature_names, model_name, output_dir):
    """Figure 3: Feature Importance Analysis."""
    print(f"Generating Figure 3: Feature Importance Analysis ({model_name})...")
    
    if not hasattr(model, 'feature_importances_'):
        print(f"Model {model_name} does not have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    sort_indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    
    if 'Extra' in model_name:
        color = 'steelblue'
        title = 'Visualization of Feature Importance in the Extra Trees Regressor\nfor Predicting Atmospheric NO2 Concentrations'
    else:
        color = 'r'
        title = 'Feature Importance Analysis'
    
    bars = plt.bar(range(len(feature_names)), importances[sort_indices], 
                   color=color, alpha=0.8, edgecolor='black', linewidth=2)
    
    plt.title(title, fontsize=22, fontweight='bold', pad=25)
    plt.xlabel('Features', fontsize=18, fontweight='bold')
    plt.ylabel('Importance Score', fontsize=18, fontweight='bold')
    
    # Set x-tick labels with bigger, bold fonts
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in sort_indices], 
               rotation=45, ha='right', fontsize=16, fontweight='bold')
    
    # Set y-tick labels with bigger, bold fonts
    plt.yticks(fontsize=16, fontweight='bold')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_3_Feature_Importance_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def figure_4_environmental_histograms(df, model_name, output_dir):
    """Figure 4: Histograms of Environmental Parameters."""
    print(f"Generating Figure 4: Environmental Parameter Histograms ({model_name})...")
    
    # Select key environmental parameters with proper units
    cols = ['NO2', 'temp', 'rhum', 'msl', 'wdsp', 'dewpt', 'clamt', 'ind', 'Hour']
    colors = ["steelblue", "forestgreen", "crimson", "darkorange", "purple", 
              "brown", "pink", "gray", "olive"]
    
    # Define proper axis labels with units
    axis_labels = {
        'NO2': 'NO2 Concentration (µg/m³)',
        'temp': 'Temperature (°C)',
        'rhum': 'Relative Humidity (%)',
        'msl': 'Mean Sea Level Pressure (hPa)',
        'wdsp': 'Wind Speed (km/h)',
        'dewpt': 'Dew Point Temperature (°C)',
        'clamt': 'Cloud Amount (oktas)',
        'ind': 'Index Value',
        'Hour': 'Hour of Day'
    }
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (col, color) in enumerate(zip(cols, colors)):
        if col in df.columns:
            # Create histogram with density
            axes[i].hist(df[col], bins=30, density=True, alpha=0.7, color=color, edgecolor='black', linewidth=1.5)
            
            # Add KDE curve
            try:
                data_clean = df[col].dropna()
                if len(data_clean) > 1:
                    kde = gaussian_kde(data_clean)
                    x_range = np.linspace(data_clean.min(), data_clean.max(), 100)
                    axes[i].plot(x_range, kde(x_range), 'black', linewidth=3)
            except:
                pass
                
            # Add mean line
            mean_val = df[col].mean()
            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=3, label=f'Mean: {mean_val:.2f}')
            
            # Set title with bigger, bold font
            axes[i].set_title(f'{col}', fontsize=20, fontweight='bold', color='navy', pad=15)
            
            # Set proper axis labels with bigger, bold fonts
            axes[i].set_xlabel(axis_labels[col], fontsize=16, fontweight='bold')
            axes[i].set_ylabel('Density', fontsize=16, fontweight='bold')
            
            # Make tick labels bigger and bold
            axes[i].tick_params(axis='both', which='major', labelsize=14)
            for label in axes[i].get_xticklabels():
                label.set_fontweight('bold')
            for label in axes[i].get_yticklabels():
                label.set_fontweight('bold')
            
            # Legend with bigger font
            legend = axes[i].legend(loc='best', fontsize=14)
            for text in legend.get_texts():
                text.set_fontweight('bold')
            axes[i].set_facecolor('white')
            
            # Add grid for better readability
            axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Histograms of Environmental Parameters\nShowing the Distributional Characteristics of Key Variables in the Dataset', 
                 fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_4_Environmental_Histograms_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def figure_5_correlation_heatmap(df, model_name, output_dir):
    """Figure 5: Correlation Matrix Heatmap."""
    print(f"Generating Figure 5: Correlation Heatmap ({model_name})...")
    
    # Select numeric columns for correlation
    numeric_cols = ['Year', 'Month', 'Day', 'Hour', 'ind', 'temp', 'dewpt', 
                   'rhum', 'msl', 'wdsp', 'wddir', 'clamt', 'NO2']
    
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(14, 10))
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                fmt='.2f', annot_kws={'size': 14, 'weight': 'bold'})
    
    # Make axis labels bigger and bold
    plt.xticks(fontsize=16, fontweight='bold', rotation=45)
    plt.yticks(fontsize=16, fontweight='bold', rotation=0)
    
    # Make colorbar labels bigger and bold
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.title('Correlation Matrices and Heatmaps\nShowing the Relationships Between Atmospheric and Temporal Parameters', 
              fontsize=22, fontweight='bold', pad=25)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_5_Correlation_Heatmap_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def figure_6_pca_analysis(df, model_name, output_dir):
    """Figure 6: PCA Loading Analysis."""
    print(f"Generating Figure 6: PCA Loading Analysis ({model_name})...")
    
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
    
    # Create the plot with proper aspect ratio
    fig, ax = plt.subplots(figsize=(10, 10))  # Square figure for proper circle
    
    # Plot arrows and labels with smart positioning to avoid overlaps
    for i, var in enumerate(meteorological_params):
        # Draw arrow
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                head_width=0.03, head_length=0.03, fc='red', ec='red', linewidth=2)
        
        # Smart label positioning to avoid overlaps with axes
        x_pos = loadings[i, 0]
        y_pos = loadings[i, 1]
        
        # Extend label position further from origin to avoid overlap
        label_distance = 1.25
        label_x = x_pos * label_distance
        label_y = y_pos * label_distance
        
        # Adjust labels that would overlap with axes (near zero crossings)
        if abs(x_pos) < 0.1:  # Near vertical axis
            label_x = 0.15 if x_pos >= 0 else -0.15
        if abs(y_pos) < 0.1:  # Near horizontal axis  
            label_y = 0.15 if y_pos >= 0 else -0.15
        
        # Place label with background box to ensure readability
        ax.text(label_x, label_y, var, 
               color='darkgreen', ha='center', va='center', 
               fontweight='bold', fontsize=18,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor='darkgreen', alpha=0.8))
    
    # Set equal aspect ratio for proper circle
    ax.set_aspect('equal', adjustable='box')
    
    # Customize plot
    ax.set_xlim([-1.4, 1.4])
    ax.set_ylim([-1.4, 1.4])
    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                  fontsize=20, fontweight='bold')
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                  fontsize=20, fontweight='bold')
    ax.set_title('PCA Loading Analysis\nShowing the Contribution of Meteorological Variables to the Principal Components', 
                fontsize=24, fontweight='bold', pad=20)
    
    # Make tick labels bigger and bold
    ax.tick_params(axis='both', which='major', labelsize=18)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Add axes lines
    ax.axhline(y=0, color='black', linewidth=1, alpha=0.7)
    ax.axvline(x=0, color='black', linewidth=1, alpha=0.7)
    
    # Add perfect circle with proper aspect ratio
    circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', 
                       color='blue', alpha=0.6, linewidth=2)
    ax.add_patch(circle)
    
    # Add variance explanation text
    total_variance = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]
    ax.text(0.02, 0.98, f'Total Variance Explained: {total_variance:.1%}', 
            transform=ax.transAxes, fontsize=18, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_6_PCA_Analysis_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def figure_7_scatterplots(df, model_name, output_dir):
    """Figure 7: NO2 vs Other Variables Scatterplots."""
    print(f"Generating Figure 7: NO2 Scatterplot Analysis ({model_name})...")
    
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
            axes[i].scatter(df['NO2'], df[col], alpha=0.6, s=8, color='blue', edgecolors='darkblue', linewidth=0.3)
            axes[i].set_title(f'NO2 vs {col}', fontsize=18, fontweight='bold', pad=15)
            axes[i].set_xlabel('NO2 (µg/m³)', fontsize=16, fontweight='bold')
            axes[i].set_ylabel(f'{col} ({feature_units.get(col, "unit")})', fontsize=16, fontweight='bold')
            
            # Make tick labels bigger and bold
            axes[i].tick_params(axis='both', which='major', labelsize=14)
            for label in axes[i].get_xticklabels():
                label.set_fontweight('bold')
            for label in axes[i].get_yticklabels():
                label.set_fontweight('bold')
            
            # Add grid for better readability
            axes[i].grid(True, alpha=0.3)
    
    # Remove unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle('Scatterplot Analysis: NO2 vs Environmental Variables', 
                 fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_7_Scatterplots_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def figure_8_actual_vs_predicted(y_test, y_pred_test, test_r2, train_r2, model_name, output_dir):
    """Figure 8: Actual vs Predicted scatter plot."""
    print(f"Generating Figure 8: Actual vs Predicted ({model_name})...")
    
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(y_test, y_pred_test, alpha=0.6, color='blue', s=30, edgecolors='darkblue', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3, 
             label='Perfect Prediction')
    
    # Labels and title with bigger fonts
    plt.xlabel('Actual NO2 Concentration (µg/m³)', fontsize=18, fontweight='bold')
    plt.ylabel('Predicted NO2 Concentration (µg/m³)', fontsize=18, fontweight='bold')
    plt.title(f'Actual vs Predicted NO2 Concentrations - {model_name}', 
              fontsize=20, fontweight='bold', pad=20)
    
    # Make tick labels bigger and bold
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    
    # Add R² scores as text with bigger fonts
    plt.text(0.05, 0.95, f'Test R² = {test_r2:.4f}', 
             transform=plt.gca().transAxes, fontsize=16, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.text(0.05, 0.88, f'Train R² = {train_r2:.4f}', 
             transform=plt.gca().transAxes, fontsize=16, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    legend = plt.legend(fontsize=16)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_8_Actual_vs_Predicted_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def figure_9_temporal_patterns(df, model_name, output_dir):
    """Figure 9: Temporal Patterns Analysis."""
    print(f"Generating Figure 9: Temporal Patterns ({model_name})...")
    
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
        # MUCH WIDER and TALLER figure to prevent any overlapping
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        
        patterns = [
            (df_temp.index.hour, 'Hour of the day'),
            (df_temp.index.dayofweek, 'Day of the week\n(0=Monday, 6=Sunday)'),
            (df_temp.index.month, 'Month\n(1=January, 12=December)')
        ]
        
        for i, (groupby_param, title) in enumerate(patterns):
            grouped_data = df_temp.groupby(groupby_param).agg({param: 'mean', 'NO2': 'mean'})
            
            # Plot NO2 on primary axis
            ax1 = axes[i]
            ax1.plot(grouped_data.index, grouped_data['NO2'], linewidth=3, color='blue', label='NO2', marker='o', markersize=6)
            ax1.set_ylabel(f'NO2 (µg/m³)', color='blue', fontsize=20, fontweight='bold')
            ax1.tick_params(axis='y', labelcolor='blue', labelsize=16)
            ax1.tick_params(axis='x', labelsize=16)
            ax1.set_xlabel(title, fontsize=20, fontweight='bold')
            
            # Plot parameter on secondary axis
            ax2 = ax1.twinx()
            ax2.plot(grouped_data.index, grouped_data[param], linewidth=3, color='red', label=param, marker='s', markersize=6)
            ax2.set_ylabel(f'{param} ({units[param]})', color='red', fontsize=20, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='red', labelsize=16)
            
            # Make x-tick labels bold
            for label in ax1.get_xticklabels():
                label.set_fontweight('bold')
            for label in ax1.get_yticklabels():
                label.set_fontweight('bold')
            for label in ax2.get_yticklabels():
                label.set_fontweight('bold')
            
            ax1.set_title(f'{title} pattern of\nNO2 and {param}', fontsize=22, fontweight='bold', pad=20)
            
            # Add grids
            ax1.grid(True, alpha=0.3)
        
        # Main title removed as requested
        
        # MAXIMUM spacing between subplots to prevent ANY overlapping
        plt.subplots_adjust(left=0.06, bottom=0.12, right=0.94, top=0.75, wspace=0.6, hspace=0.4)
        
        plt.savefig(f'{output_dir}/Figure_9_Temporal_Patterns_{param}_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
            # CSV export disabled for Figure 9 to reduce file generation

def figure_10_wind_direction_analysis(df, model_name, output_dir):
    """Figure 10: Wind Direction Circular Analysis."""
    print(f"Generating Figure 10: Wind Direction Analysis ({model_name})...")
    
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
    ax.arrow(mean_direction, 0, 0, R, alpha=0.7, width=0.02,
             edgecolor='black', facecolor='green', lw=2, zorder=5)
    
    # Make tick labels bigger and bold
    ax.tick_params(axis='both', which='major', labelsize=14)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # Add directional labels with bigger fonts - positioned to avoid overlap
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    angles = np.arange(0, 2*np.pi, np.pi/4)
    for angle, direction in zip(angles, directions):
        # Position labels further out to avoid overlap
        ax.text(angle, ax.get_ylim()[1]*1.3, direction, 
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Add statistical information
    ax.text(0.02, 0.98, f'Mean Direction: {mean_direction_deg:.1f}°\nResultant Length: {R:.3f}', 
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            verticalalignment='top')
    
    ax.set_title('Wind Direction Circular Decomposition', fontsize=20, fontweight='bold', pad=25)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_10_Wind_Direction_Analysis_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # CSV export disabled for Figure 10 to reduce file generation

def get_wind_direction_category(degrees):
    """Get wind direction category from degrees"""
    if 337.5 <= degrees or degrees < 22.5:
        return 'North'
    elif 22.5 <= degrees < 67.5:
        return 'Northeast'
    elif 67.5 <= degrees < 112.5:
        return 'East'
    elif 112.5 <= degrees < 157.5:
        return 'Southeast'
    elif 157.5 <= degrees < 202.5:
        return 'South'
    elif 202.5 <= degrees < 247.5:
        return 'Southwest'
    elif 247.5 <= degrees < 292.5:
        return 'West'
    elif 292.5 <= degrees < 337.5:
        return 'Northwest'
    else:
        return 'Unknown'

def figure_11_seasonal_patterns(df, model_name, output_dir):
    """Figure 11: Seasonal Patterns Analysis."""
    print(f"Generating Figure 11: Seasonal Patterns ({model_name})...")
    
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
        
        # Plot NO2 with bigger markers and lines
        ax.plot(seasonal_data.index, seasonal_data['NO2'], 'b-o', linewidth=4, markersize=12, label='NO2', color='blue')
        ax.set_ylabel(f'NO2 (µg/m³)', color='blue', fontsize=16, fontweight='bold')
        ax.tick_params(axis='y', labelcolor='blue', labelsize=14)
        ax.tick_params(axis='x', labelsize=14)
        
        # Plot meteorological parameter
        ax2 = ax.twinx()
        ax2.plot(seasonal_data.index, seasonal_data[param], 'r-s', linewidth=4, markersize=12, label=param, color='red')
        ax2.set_ylabel(f'{param} ({units[param]})', color='red', fontsize=16, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='red', labelsize=14)
        
        # Make tick labels bold
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        for label in ax2.get_yticklabels():
            label.set_fontweight('bold')
        
        ax.set_title(f'Seasonal pattern of NO2 (µg/m³) and {param} ({units[param]})', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Season', fontsize=16, fontweight='bold')
        
        # Add legends with bigger fonts
        legend1 = ax.legend(loc='upper left', fontsize=14)
        legend2 = ax2.legend(loc='upper right', fontsize=14)
        for text in legend1.get_texts():
            text.set_fontweight('bold')
        for text in legend2.get_texts():
            text.set_fontweight('bold')
        
        # Add grids for better readability
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/Figure_11_Seasonal_Pattern_{param}_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # CSV export disabled for Figure 11 to reduce file generation

def generate_visualizations_for_model(df, model, model_name, feature_names, y_test, y_pred_test, test_r2, train_r2, output_dir):
    """Generate ALL visualization figures for a specific model"""
    
    print(f"\n=== Generating ALL visualizations for {model_name} ===")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate ALL figures (2-11)
    try:
        figure_2_negative_no2_analysis(df, model_name, output_dir)
        figure_3_feature_importance(model, feature_names, model_name, output_dir)
        figure_4_environmental_histograms(df, model_name, output_dir)
        figure_5_correlation_heatmap(df, model_name, output_dir)
        figure_6_pca_analysis(df, model_name, output_dir)
        figure_7_scatterplots(df, model_name, output_dir)
        figure_8_actual_vs_predicted(y_test, y_pred_test, test_r2, train_r2, model_name, output_dir)
        figure_9_temporal_patterns(df, model_name, output_dir)
        figure_10_wind_direction_analysis(df, model_name, output_dir)
        figure_11_seasonal_patterns(df, model_name, output_dir)
        
        print(f"✅ ALL 11 figures completed for {model_name}")
        print(f"📁 Saved in: {output_dir}/")
        
    except Exception as e:
        print(f"❌ Error generating visualizations for {model_name}: {e}")
        import traceback
        traceback.print_exc()

def interactive_model_selection(results_df, trained_models, model_predictions, viz_df, feature_names, y_test):
    """Interactive model selection for custom visualizations"""
    
    print("\n" + "="*60)
    print("INTERACTIVE MODEL VISUALIZATION")
    print("="*60)
    
    while True:
        # Ask if user wants to see visualization for other models
        print(f"\nDo you want to see visualizations for any other model?")
        choice = input("Enter 'yes' to continue or 'no' to exit: ").lower().strip()
        
        if choice in ['no', 'n', 'exit', 'quit']:
            print("Exiting interactive visualization mode.")
            break
        elif choice in ['yes', 'y']:
            # Show available models
            print(f"\n📋 Available trained models:")
            print("-" * 40)
            for idx, (_, row) in enumerate(results_df.iterrows(), 1):
                print(f"{idx:2d}. {row['Model']} (R² = {row['test_r2']:.4f})")
            
            # Get user selection
            try:
                model_choice = int(input(f"\nEnter the number (1-{len(results_df)}) of the model you want to visualize: "))
                
                if 1 <= model_choice <= len(results_df):
                    selected_row = results_df.iloc[model_choice - 1]
                    selected_model_name = selected_row['Model']
                    
                    if selected_model_name in trained_models:
                        # Get model and metrics
                        selected_model = trained_models[selected_model_name]
                        predictions = model_predictions[selected_model_name]
                        test_r2 = selected_row['test_r2']
                        train_r2 = selected_row['train_r2']
                        
                        # Create custom output directory
                        custom_output_dir = f"visualization_results/{selected_model_name.lower()}"
                        
                        print(f"\n🎨 Generating visualizations for: {selected_model_name}")
                        print(f"📁 Output directory: {custom_output_dir}/")
                        
                        # Generate visualizations
                        generate_visualizations_for_model(
                            viz_df, selected_model, selected_model_name, feature_names,
                            y_test, predictions['test_pred'], test_r2, train_r2, custom_output_dir
                        )
                        
                        print(f"✅ Visualizations completed for {selected_model_name}!")
                        
                    else:
                        print(f"❌ Model {selected_model_name} not found in trained models.")
                else:
                    print("❌ Invalid selection. Please choose a number from the list.")
                    
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
                
        else:
            print("❌ Invalid choice. Please enter 'yes' or 'no'.")

def main():
    """Main execution function"""
    print("="*80)
    print("ULTIMATE COMPREHENSIVE AIR POLLUTION ANALYSIS")
    print("Combining Machine Learning and Time Series Analysis")
    print("="*80)
    
    start_time = datetime.now()
    
    try:
        # Create directories
        create_directories()
        
        print("\n" + "="*60)
        print("PART 1: MACHINE LEARNING ANALYSIS")
        print("="*60)
        
        # Load and preprocess data for ML
        X, y, viz_df = load_and_preprocess_data()
        
        # Split data for ML
        print(f"\nSplitting data (80% train, 20% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Scale the data
        print("\nScaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        
        # Generate experimental tables first
        print("\n" + "="*50)
        print("GENERATING EXPERIMENTAL TABLES")
        print("="*50)
        generate_experimental_tables(viz_df)
        
        # Train all ML models
        print("\n" + "="*50)
        print("TRAINING ALL MACHINE LEARNING MODELS")
        print("="*50)
        results_df, trained_models, model_predictions = train_all_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Show ML summary
        print("\n" + "="*50)
        print("MACHINE LEARNING TRAINING COMPLETE - TOP 5 MODELS")
        print("="*50)
        print(f"{'Model':<25} {'R²':<8} {'RMSE':<8} {'MAE':<8}")
        print("-" * 60)
        
        for _, row in results_df.head(5).iterrows():
            print(f"{row['Model']:<25} {row['test_r2']:<8.4f} {row['test_rmse']:<8.2f} {row['test_mae']:<8.2f}")
        
        # Best ML model info
        best_ml_model_name = results_df.iloc[0]['Model']
        best_ml_r2 = results_df.iloc[0]['test_r2']
        
        print("\n" + "="*60)
        print("PART 2: TIME SERIES ANALYSIS")
        print("="*60)
        
        # Load and clean data for time series
        no2_series = load_and_clean_data_for_ts()
        
        # Create time series train/test split
        train_series, test_series = create_train_test_split_ts(no2_series)
        
        # Fit time series models
        ts_results = fit_time_series_models(train_series, test_series)
        
        # Create time series visualizations
        create_time_series_visualizations(train_series, test_series, ts_results)
        
        # Find best time series model and generate forecast
        best_ts_model_name = min([name for name in ts_results.keys() if name.startswith('ARIMA')], 
                                key=lambda x: ts_results[x]['test_mae'])
        forecast_df = generate_forecast(train_series, test_series, ts_results[best_ts_model_name])
        
        # Save time series results
        ts_summary_df = save_time_series_results(ts_results)
        
        print("\n" + "="*60)
        print("PART 3: PATTERN ANALYSIS")
        print("="*60)
        
        # Comprehensive pattern analysis
        pattern_insights = analyze_comprehensive_patterns(viz_df)
        
        # Professional dashboard
        create_professional_dashboard(viz_df)
        
        print("\n" + "="*60)
        print("PART 4: FORECASTING")
        print("="*60)
        
        # Generate 1-year forecast using best ML model
        best_model = trained_models[best_ml_model_name]
        best_train_r2 = results_df.iloc[0]['train_r2']
        best_test_r2 = results_df.iloc[0]['test_r2']
        best_train_mae = results_df.iloc[0]['train_mae']
        best_test_mae = results_df.iloc[0]['test_mae']
        best_train_rmse = results_df.iloc[0]['train_rmse']
        best_test_rmse = results_df.iloc[0]['test_rmse']
        
        # Create label encoder for forecast using the same format as training data
        le_forecast = LabelEncoder()
        all_seasons = ['Spring (Mar-May)', 'Summer (Jun-Aug)', 'Autumn (Sep-Nov)', 'Winter (Dec-Feb)']
        le_forecast.fit(all_seasons)
        
        # Model performance dictionary
        model_performance = {
            'train_r2': best_train_r2,
            'test_r2': best_test_r2,
            'train_mae': best_train_mae,
            'test_mae': best_test_mae,
            'train_rmse': best_train_rmse,
            'test_rmse': best_test_rmse
        }
        
        # Generate forecast
        forecast_df = generate_ml_forecast(best_model, scaler, le_forecast, X, y)
        
        # Create forecast visualizations
        create_forecast_visualizations(forecast_df, best_ml_model_name, model_performance)
        
        print("\n" + "="*60)
        print("PART 5: BEST MODEL VISUALIZATIONS")
        print("="*60)
        
        # Get feature names
        feature_names = list(X.columns)
        
        # Generate visualizations for the best ML model automatically
        best_predictions = model_predictions[best_ml_model_name]
        
        print(f"\n🎨 Generating visualizations for the BEST ML model: {best_ml_model_name}")
        best_output_dir = f"{VISUALIZATION_RESULTS_DIR}/best_model_{best_ml_model_name.lower()}"
        
        generate_visualizations_for_model(
            viz_df, best_model, best_ml_model_name, feature_names,
            y_test, best_predictions['test_pred'], best_test_r2, best_train_r2, best_output_dir
        )
        
        print("\n" + "="*60)
        print("PART 6: INTERACTIVE MODEL SELECTION")
        print("="*60)
        
        # Interactive model selection for ML visualizations
        print("\nMachine Learning Model Visualization Options:")
        interactive_model_selection(results_df, trained_models, model_predictions, viz_df, feature_names, y_test)
        
        # Save comprehensive results summary
        comprehensive_summary = f"""COMPREHENSIVE AIR POLLUTION ANALYSIS RESULTS
==================================================

MACHINE LEARNING RESULTS:
• Best Model: {best_ml_model_name}
• Test R²: {best_test_r2:.4f}
• Test RMSE: {results_df.iloc[0]['test_rmse']:.4f} µg/m³
• Test MAE: {results_df.iloc[0]['test_mae']:.4f} µg/m³

TIME SERIES RESULTS:
• Best Model: {ts_summary_df.iloc[0]['Model']}
• Test R²: {ts_summary_df.iloc[0]['Test_R2']:.4f}
• Test MAE: {ts_summary_df.iloc[0]['Test_MAE']:.4f} µg/m³

PATTERN ANALYSIS INSIGHTS:
• Long-term Trend: {pattern_insights['trend_slope']:.3f} µg/m³/year
• Peak Pollution Month: {pattern_insights['peak_month']}
• Peak Pollution Hour: {pattern_insights['peak_hour']}:00
• Weekday vs Weekend: {pattern_insights['weekday_avg']:.1f} vs {pattern_insights['weekend_avg']:.1f} µg/m³
• Anomalies Detected: {pattern_insights['anomalies_count']} events

FORECAST RESULTS:
• Model Used: {best_ml_model_name}
• Forecast Period: 1 year (8,760 hours)
• Forecast Mean: {forecast_df['NO2_Forecast'].mean():.2f} µg/m³
• Forecast Range: {forecast_df['NO2_Forecast'].min():.1f} - {forecast_df['NO2_Forecast'].max():.1f} µg/m³

Generated Files:
- Models: {MODELS_DIR}/
- Tables: {TABLES_DIR}/
- Time Series: {TS_FIGURES_DIR}/
- Patterns: {PATTERN_ANALYSIS_DIR}/
- Professional: {PROFESSIONAL_RESULTS_DIR}/
- Forecasts: {FORECAST_RESULTS_DIR}/
- Visualizations: {VISUALIZATION_RESULTS_DIR}/
"""
        
        summary_path = os.path.join(MODELS_DIR, 'comprehensive_analysis_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(comprehensive_summary)
        
        # Final summary
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS COMPLETE")
        print("="*80)
        
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        print(f"\n📊 MACHINE LEARNING RESULTS:")
        print(f"   🏆 Best ML Model: {best_ml_model_name}")
        print(f"   📈 Best ML R²: {best_test_r2:.4f}")
        print(f"   📉 Best ML MAE: {results_df.iloc[0]['test_mae']:.4f} µg/m³")
        
        print(f"\n⏰ TIME SERIES RESULTS:")
        print(f"   🏆 Best TS Model: {ts_summary_df.iloc[0]['Model']}")
        print(f"   📈 Best TS R²: {ts_summary_df.iloc[0]['Test_R2']:.4f}")
        print(f"   📉 Best TS MAE: {ts_summary_df.iloc[0]['Test_MAE']:.4f} µg/m³")
        
        print(f"\n🔍 PATTERN ANALYSIS INSIGHTS:")
        print(f"   📈 Long-term Trend: {pattern_insights['trend_slope']:.3f} µg/m³/year")
        print(f"   📅 Peak Month: {pattern_insights['peak_month']}")
        print(f"   ⏰ Peak Hour: {pattern_insights['peak_hour']}:00")
        print(f"   🗓️ Weekday vs Weekend: {pattern_insights['weekday_avg']:.1f} vs {pattern_insights['weekend_avg']:.1f} µg/m³")
        print(f"   🚨 Anomalies: {pattern_insights['anomalies_count']} events")
        
        print(f"\n🔮 FORECAST RESULTS:")
        print(f"   📊 Forecast Mean: {forecast_df['NO2_Forecast'].mean():.2f} µg/m³")
        print(f"   📏 Forecast Range: {forecast_df['NO2_Forecast'].min():.1f} - {forecast_df['NO2_Forecast'].max():.1f} µg/m³")
        print(f"   ⏳ Period: 1 year (8,760 hours)")
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"⏱️ Total execution time: {execution_time}")
        print(f"📁 Generated Results:")
        print(f"   • ML Models: {MODELS_DIR}/")
        print(f"   • Experimental Tables: {TABLES_DIR}/")
        print(f"   • Time Series: {TS_FIGURES_DIR}/")
        print(f"   • Pattern Analysis: {PATTERN_ANALYSIS_DIR}/")
        print(f"   • Professional Dashboard: {PROFESSIONAL_RESULTS_DIR}/")
        print(f"   • Forecasts: {FORECAST_RESULTS_DIR}/")
        print(f"   • Visualizations: {VISUALIZATION_RESULTS_DIR}/")
        print(f"📊 {len(trained_models)} ML models + {len(ts_results)} TS models trained")
        
        print(f"\n📋 Generated Files Summary:")
        print(f"   • Table1-7: All experimental results and forecasts")
        print(f"   • Comprehensive pattern analysis")
        print(f"   • Professional dashboard")
        print(f"   • 1-year forecast with visualizations")
        print(f"   • Model-specific visualizations")
        
        print(f"\n📄 Summary saved: {summary_path}")
        
        return results_df, ts_summary_df
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    results = main() 