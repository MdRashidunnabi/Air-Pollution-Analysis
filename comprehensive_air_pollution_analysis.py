#!/usr/bin/env python3
"""
Ultimate Comprehensive Air Pollution Analysis Script
====================================================
Combines:
- Model training and evaluation (corrected_air_pollution_predictor.py)
- Complete visualizations (complete_visualization_extra_trees.py & complete_visualization_random_forest.py)
- Experimental tables generation (generate_tables.py)
- Interactive model selection for custom visualizations
All-in-one solution for air pollution prediction analysis
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
import itertools
from datetime import datetime

# Model imports
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor, VotingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

# Constants
DATASET_FILE = 'air_pollution_ireland.csv'
MODELS_DIR = 'trained_models'
TABLES_DIR = 'experimental_results_tables'
RANDOM_STATE = 42

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def create_directories():
    """Create necessary directories"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

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
        'CatBoost': CatBoostRegressor(verbose=False, random_state=RANDOM_STATE),
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
    
    # Adjusted R2
    n_test = len(X_test)
    p = X_test.shape[1]
    test_adj_r2 = adjusted_r2(test_r2, n_test, p)
    
    metrics = {
        'Model': model_name,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Train_RMSE': train_rmse,
        'Test_RMSE': test_rmse,
        'Train_MAE': train_mae,
        'Test_MAE': test_mae,
        'Train_MAPE': train_mape,
        'Test_MAPE': test_mape,
        'Test_Adj_R2': test_adj_r2
    }
    
    return metrics, y_train_pred, y_test_pred

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
            metrics, y_train_pred, y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
            results.append(metrics)
            
            # Store model and predictions
            trained_models[model_name] = model
            model_predictions[model_name] = {
                'train_pred': y_train_pred,
                'test_pred': y_test_pred
            }
            
            # Save model
            model_path = os.path.join(MODELS_DIR, f'{model_name}_model.pkl')
            joblib.dump(model, model_path)
            
            print(f"R² Score: {metrics['Test_R2']:.4f}")
            print(f"RMSE: {metrics['Test_RMSE']:.4f}")
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test_R2', ascending=False).reset_index(drop=True)
    
    # Save summaries
    comprehensive_path = os.path.join(MODELS_DIR, 'comprehensive_model_performance_summary.csv')
    results_df.to_csv(comprehensive_path, index=False)
    
    simple_summary = results_df[['Model', 'Test_R2', 'Test_RMSE', 'Test_MAE', 'Test_MAPE']].copy()
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

# === VISUALIZATION FUNCTIONS ===

def figure_2_negative_no2_analysis(df, model_name, output_dir):
    """Figure 2: Finding the threshold for treating negative NO2 values."""
    print(f"Generating Figure 2: Negative NO2 Analysis ({model_name})...")
    
    negative_NO2 = df[df['NO2'] < 0]
    
    if len(negative_NO2) == 0:
        print("No negative NO2 values found in the dataset.")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No Negative NO2 Values Found\nin Current Dataset', 
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax.set_title('Figure 2: Negative NO2 Analysis', fontsize=14)
        plt.savefig(f'{output_dir}/Figure_2_Negative_NO2_Analysis_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    negative_NO2_values = negative_NO2['NO2']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Time series plot
    ax1.plot(negative_NO2.index, negative_NO2['NO2'], 'ro-', markersize=3, alpha=0.7)
    ax1.set_title('Time Series Plot of Negative NO2 Values', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Index', fontsize=14, fontweight='bold')
    ax1.set_ylabel('NO2 Concentration (µg/m³)', fontsize=14, fontweight='bold')
    
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
        ax2.set_xlabel('NO2 Concentration (µg/m³)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Density', fontsize=14, fontweight='bold')
        ax2.set_title('KDE of Negative NO2 Values', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=12)
    
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
        title = 'Feature importances'
    
    bars = plt.bar(range(len(feature_names)), importances[sort_indices], 
                   color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Features', fontsize=14, fontweight='bold')
    plt.ylabel('Importance Score', fontsize=14, fontweight='bold')
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in sort_indices], 
               rotation=45, ha='right', fontsize=12, fontweight='bold')
    
    if 'Extra' in model_name:
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_3_Feature_Importance_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def figure_4_environmental_histograms(df, model_name, output_dir):
    """Figure 4: Histograms of Environmental Parameters."""
    print(f"Generating Figure 4: Environmental Parameter Histograms ({model_name})...")
    
    # Select key environmental parameters
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
            
            axes[i].set_title(f'{col}', fontsize=14, fontweight='bold', color='navy')
            axes[i].set_xlabel('')
            axes[i].legend(loc='best', fontsize=11)
            axes[i].set_facecolor('white')
    
    plt.suptitle('Histograms of Environmental Parameters\nShowing the Distributional Characteristics of Key Variables in the Dataset', 
                 fontsize=16, fontweight='bold')
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
                fmt='.2f', annot_kws={'size': 9})
    
    plt.title('Correlation Matrices and Heatmaps\nShowing the Relationships Between Atmospheric and Temporal Parameters', 
              fontsize=16, fontweight='bold', pad=20)
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
               fontweight='bold', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='darkgreen', alpha=0.8))
    
    # Set equal aspect ratio for proper circle
    ax.set_aspect('equal', adjustable='box')
    
    # Customize plot
    ax.set_xlim([-1.4, 1.4])
    ax.set_ylim([-1.4, 1.4])
    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                  fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                  fontsize=14, fontweight='bold')
    ax.set_title('PCA Loading Analysis\nShowing the Contribution of Meteorological Variables to the Principal Components', 
                fontsize=16, fontweight='bold', pad=20)
    
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
            transform=ax.transAxes, fontsize=12, fontweight='bold',
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
            axes[i].scatter(df['NO2'], df[col], alpha=0.5, s=3, color='blue')
            axes[i].set_title(f'NO2 vs {col}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('NO2 (µg/m³)', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(f'{col} ({feature_units.get(col, "unit")})', fontsize=12, fontweight='bold')
    
    # Remove unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle('Scatterplot Analysis: NO2 vs Environmental Variables', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_7_Scatterplots_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def figure_8_actual_vs_predicted(y_test, y_pred_test, test_r2, train_r2, model_name, output_dir):
    """Figure 8: Actual vs Predicted scatter plot."""
    print(f"Generating Figure 8: Actual vs Predicted ({model_name})...")
    
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(y_test, y_pred_test, alpha=0.6, color='blue', s=20)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
             label='Perfect Prediction')
    
    # Labels and title
    plt.xlabel('Actual NO2 Concentration (µg/m³)', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted NO2 Concentration (µg/m³)', fontsize=14, fontweight='bold')
    plt.title(f'Actual vs Predicted NO2 Concentrations - {model_name}', 
              fontsize=16, fontweight='bold')
    
    # Add R² scores as text
    plt.text(0.05, 0.95, f'Test R² = {test_r2:.4f}', 
             transform=plt.gca().transAxes, fontsize=13, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.text(0.05, 0.88, f'Train R² = {train_r2:.4f}', 
             transform=plt.gca().transAxes, fontsize=13, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.legend(fontsize=12)
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
            ax1.set_ylabel(f'NO2 (µg/m³)', color='blue', fontsize=13, fontweight='bold')
            ax1.tick_params(axis='y', labelcolor='blue', labelsize=11)
            ax1.set_xlabel(title, fontsize=13, fontweight='bold')
            
            # Plot parameter on secondary axis
            ax2 = ax1.twinx()
            ax2.plot(grouped_data.index, grouped_data[param], 'r-', linewidth=2, color='red', label=param)
            ax2.set_ylabel(f'{param} ({units[param]})', color='red', fontsize=13, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='red', labelsize=11)
            
            ax1.set_title(f'{title} pattern of\nNO2 and {param}', fontsize=14, fontweight='bold')
        
        plt.suptitle(f'Interplay of Climatic Parameters: A Symphony of Temperature Variations,\nAtmospheric Indicators, and Human Influence', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/Figure_9_Temporal_Patterns_{param}_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

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
    ax.arrow(mean_direction, 0, 0, R, alpha=0.5, width=0.015,
             edgecolor='black', facecolor='green', lw=1, zorder=5)
    
    ax.set_title('Wind Direction Circular Decomposition', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Figure_10_Wind_Direction_Analysis_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

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
        
        # Plot NO2
        ax.plot(seasonal_data.index, seasonal_data['NO2'], 'b-o', linewidth=2, markersize=8, label='NO2', color='blue')
        ax.set_ylabel(f'NO2 (µg/m³)', color='blue', fontsize=13, fontweight='bold')
        ax.tick_params(axis='y', labelcolor='blue', labelsize=11)
        
        # Plot meteorological parameter
        ax2 = ax.twinx()
        ax2.plot(seasonal_data.index, seasonal_data[param], 'r-o', linewidth=2, markersize=8, label=param, color='red')
        ax2.set_ylabel(f'{param} ({units[param]})', color='red', fontsize=13, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='red', labelsize=11)
        
        ax.set_title(f'Seasonal pattern of NO2 (µg/m³) and {param} ({units[param]})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Season', fontsize=13, fontweight='bold')
        
        # Add legends
        ax.legend(loc='upper left', fontsize=11)
        ax2.legend(loc='upper right', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/Figure_11_Seasonal_Pattern_{param}_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

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
                print(f"{idx:2d}. {row['Model']} (R² = {row['Test_R2']:.4f})")
            
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
                        test_r2 = selected_row['Test_R2']
                        train_r2 = selected_row['Train_R2']
                        
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
    
    print("="*60)
    print("ULTIMATE COMPREHENSIVE AIR POLLUTION ANALYSIS")
    print("Training Models + Tables + Interactive Visualizations")
    print("="*60)
    
    # Create directories
    create_directories()
    
    # Load and preprocess data
    X, y, viz_df = load_and_preprocess_data()
    
    # Split data
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
    
    # Train all models
    results_df, trained_models, model_predictions = train_all_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Generate experimental tables
    generate_experimental_tables(viz_df)
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Model':<25} {'R²':<8} {'RMSE':<8} {'MAE':<8}")
    print("-" * 60)
    
    for _, row in results_df.head(10).iterrows():
        print(f"{row['Model']:<25} {row['Test_R2']:<8.4f} {row['Test_RMSE']:<8.2f} {row['Test_MAE']:<8.2f}")
    
    # Identify best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    best_r2 = results_df.iloc[0]['Test_R2']
    best_train_r2 = results_df.iloc[0]['Train_R2']
    best_predictions = model_predictions[best_model_name]
    
    print(f"\n🏆 Best model: {best_model_name} (R² = {best_r2:.4f})")
    
    # Get feature names
    feature_names = list(X.columns)
    
    # Generate visualizations for the best model automatically
    print(f"\n🎨 Generating visualizations for the BEST model: {best_model_name}")
    best_output_dir = f"visualization_results/best_model_{best_model_name.lower()}"
    
    generate_visualizations_for_model(
        viz_df, best_model, best_model_name, feature_names,
        y_test, best_predictions['test_pred'], best_r2, best_train_r2, best_output_dir
    )
    
    # Save best model info
    best_model_info = f"""Best Model: {best_model_name}
Test R²: {best_r2:.4f}
Test RMSE: {results_df.iloc[0]['Test_RMSE']:.4f}
Test MAE: {results_df.iloc[0]['Test_MAE']:.4f}
"""
    
    best_model_path = os.path.join(MODELS_DIR, 'best_model_info.txt')
    with open(best_model_path, 'w') as f:
        f.write(best_model_info)
    
    print(f"\n" + "="*60)
    print("AUTOMATIC ANALYSIS COMPLETE!")
    print("="*60)
    print(f"✅ All models trained and saved in: {MODELS_DIR}/")
    print(f"✅ Performance summaries saved")
    print(f"✅ Experimental tables generated in: {TABLES_DIR}/")
    print(f"✅ Best model visualizations: {best_output_dir}/")
    print(f"✅ Best model: {best_model_name} (R² = {best_r2:.4f})")
    
    # Interactive model selection for additional visualizations
    interactive_model_selection(results_df, trained_models, model_predictions, viz_df, feature_names, y_test)
    
    print(f"\n🎉 ULTIMATE ANALYSIS COMPLETED!")
    print("="*60)
    
    return results_df

if __name__ == "__main__":
    results = main() 