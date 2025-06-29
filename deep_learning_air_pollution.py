#!/usr/bin/env python3
"""
Deep Learning Air Pollution Analysis
====================================
Advanced deep learning models for air pollution prediction using Ireland dataset.
Includes 12 different deep learning architectures with proper error handling.
"""

import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

# Suppress TensorFlow warnings and configure environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# Core libraries
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
from scikeras.wrappers import KerasRegressor
import torch
import torch.nn as nn
import torch.optim as optim

# Configure TensorFlow for CPU priority with GPU fallback
try:
    # Force CPU-only execution to avoid CUDA libdevice issues
    tf.config.set_visible_devices([], 'GPU')
    print("ℹ️  Configured TensorFlow for CPU-only execution")
except Exception as e:
    print(f"⚠️  TensorFlow configuration warning: {e}")

# Global configuration
DATASET_FILE = 'air_pollution_ireland.csv'
RESULTS_CSV = 'experimental_results_tables/Table8_Deep_Learning_Models_Performance.csv'
RANDOM_STATE = 42
EPOCHS = 30
BATCH_SIZE = 32

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

def calculate_rmse(y_true, y_pred):
    """Calculate RMSE using numpy to avoid scikit-learn compatibility issues"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mape(y_true, y_pred):
    """Calculate MAPE with proper handling of zero values"""
    y_true_safe = np.where(np.abs(y_true) < 1e-8, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

def map_season(month):
    """Map month to season for feature engineering"""
    if month in [3, 4, 5]: return 0  # Spring
    if month in [6, 7, 8]: return 1  # Summer
    if month in [9, 10, 11]: return 2  # Autumn
    return 3  # Winter

def load_and_preprocess_data():
    """Load and preprocess the air pollution dataset"""
    print("📊 Loading and preprocessing dataset...")
    
    try:
        df = pd.read_csv(DATASET_FILE)
        print(f"   Original dataset shape: {df.shape}")
        
        # Date feature engineering
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['Season'] = df['Month'].apply(map_season)
        
        # Remove unnecessary columns
        columns_to_drop = ['Date', 'rain', 'ind.1', 'ind.2', 'ind.3', 'ind.4', 
                          'wetb', 'vappr', 'ww', 'w', 'sun', 'vis', 'clht']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        print(f"   Shape after column removal: {df.shape}")
        
        # Handle negative NO2 values
        initial_rows = len(df)
        negative_no2 = df[df['NO2'] < 0]
        if len(negative_no2) > 0:
            threshold = np.percentile(negative_no2['NO2'], 5)
            print(f"   Cleaning {len(negative_no2)} negative NO2 values (threshold: {threshold:.2f})")
            df = df[df['NO2'] >= threshold]
            between_mask = (df['NO2'] < 0) & (df['NO2'] >= threshold)
            df.loc[between_mask, 'NO2'] = 0
        
        # Remove missing values
        df = df.dropna(subset=['NO2'])
        print(f"   Final dataset shape: {df.shape}")
        
        # Prepare features and target
        X = df.drop(['NO2'], axis=1)
        y = df['NO2']
        
        # Encode categorical variables
        le = LabelEncoder()
        if 'Season' in X.columns:
            X['Season'] = le.fit_transform(X['Season'])
        
        return X, y
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

# =====================
# Keras Model Architectures
# =====================

def create_dnn_model(input_dim):
    """Basic Deep Neural Network"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_deep_dnn_model(input_dim):
    """Deeper Neural Network with Batch Normalization"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_lstm_model(input_dim):
    """LSTM Network for sequence modeling"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, input_dim)),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_gru_model(input_dim):
    """GRU Network"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, input_dim)),
        tf.keras.layers.GRU(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GRU(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_bilstm_model(input_dim):
    """Bidirectional LSTM Network"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, input_dim)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_conv1d_model(input_dim):
    """1D Convolutional Network"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, input_dim)),
        tf.keras.layers.Conv1D(128, kernel_size=1, activation='relu'),
        tf.keras.layers.Conv1D(64, kernel_size=1, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_attention_lstm_model(input_dim):
    """LSTM with Attention Mechanism"""
    inputs = tf.keras.layers.Input(shape=(1, input_dim))
    lstm_out = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    
    # Attention mechanism
    attention = tf.keras.layers.Dense(1, activation='tanh')(lstm_out)
    attention = tf.keras.layers.Flatten()(attention)
    attention_weights = tf.keras.layers.Activation('softmax')(attention)
    attention_weights = tf.keras.layers.RepeatVector(128)(attention_weights)
    attention_weights = tf.keras.layers.Permute([2, 1])(attention_weights)
    
    attended = tf.keras.layers.Multiply()([lstm_out, attention_weights])
    attended = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended)
    
    output = tf.keras.layers.Dense(32, activation='relu')(attended)
    output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(1)(output)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_residual_dnn_model(input_dim):
    """Residual Deep Neural Network"""
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    
    # First residual block
    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    residual = x
    
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, residual])
    
    # Second residual block
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    residual = x
    
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, residual])
    
    # Output layers
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_wide_deep_model(input_dim):
    """Wide & Deep Network"""
    # Wide component (linear)
    wide_input = tf.keras.layers.Input(shape=(input_dim,))
    wide = tf.keras.layers.Dense(1, use_bias=False)(wide_input)
    
    # Deep component
    deep_input = tf.keras.layers.Input(shape=(input_dim,))
    deep = tf.keras.layers.Dense(256, activation='relu')(deep_input)
    deep = tf.keras.layers.Dropout(0.3)(deep)
    deep = tf.keras.layers.Dense(128, activation='relu')(deep)
    deep = tf.keras.layers.Dropout(0.2)(deep)
    deep = tf.keras.layers.Dense(64, activation='relu')(deep)
    deep = tf.keras.layers.Dense(1)(deep)
    
    # Combine wide and deep
    output = tf.keras.layers.Add()([wide, deep])
    
    model = tf.keras.Model(inputs=[wide_input, deep_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_autoencoder_model(input_dim):
    """Autoencoder with Regression Head"""
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    
    # Encoder
    encoded = tf.keras.layers.Dense(128, activation='relu')(inputs)
    encoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(32, activation='relu')(encoded)
    
    # Decoder
    decoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(128, activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(input_dim, activation='linear')(decoded)
    
    # Regression head from encoded features
    regression = tf.keras.layers.Dense(16, activation='relu')(encoded)
    regression = tf.keras.layers.Dense(1)(regression)
    
    model = tf.keras.Model(inputs=inputs, outputs=regression)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# =====================
# PyTorch Models
# =====================

class TorchMLP(nn.Module):
    """PyTorch Multi-Layer Perceptron"""
    def __init__(self, input_dim):
        super(TorchMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

class TorchResNet(nn.Module):
    """PyTorch ResNet for Tabular Data"""
    def __init__(self, input_dim):
        super(TorchResNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, 256)
        self.res_block1 = self._make_res_block(256, 256)
        self.res_block2 = self._make_res_block(256, 128)
        self.output_layer = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def _make_res_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.relu(self.input_layer(x))
        
        # First residual block
        residual = x
        x = self.res_block1(x)
        if x.shape == residual.shape:
            x = x + residual
        
        # Second residual block with dimension change
        x = self.dropout(x)
        x = self.res_block2(x)
        
        return self.output_layer(x)

class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible PyTorch MLP"""
    def __init__(self, input_dim=None, epochs=30, batch_size=32, lr=0.001, verbose=0):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self.model = None
    
    def fit(self, X, y):
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        
        self.model = TorchMLP(self.input_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        return self
    
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy().flatten()
        return predictions

class TorchResNetRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible PyTorch ResNet"""
    def __init__(self, input_dim=None, epochs=30, batch_size=32, lr=0.001, verbose=0):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self.model = None
    
    def fit(self, X, y):
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        
        self.model = TorchResNet(self.input_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        return self
    
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy().flatten()
        return predictions

# =====================
# Wrapper Classes for 3D Input Models
# =====================

class SequenceModelWrapper(BaseEstimator, RegressorMixin):
    """Generic wrapper for sequence models that need 3D input"""
    def __init__(self, model_func, input_dim=None, epochs=30, batch_size=32, verbose=0):
        self.model_func = model_func
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
    
    def fit(self, X, y):
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        
        self.model = self.model_func(self.input_dim)
        X_reshaped = np.expand_dims(X, axis=1)  # Add timestep dimension
        
        self.model.fit(
            X_reshaped, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_split=0.1
        )
        return self
    
    def predict(self, X):
        X_reshaped = np.expand_dims(X, axis=1)
        return self.model.predict(X_reshaped, verbose=0).flatten()

# =====================
# Model Evaluation
# =====================

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate model performance with comprehensive metrics"""
    try:
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        return {
            'Model_Name': model_name,
            'Model_Type': model_name.replace('_', ' '),
            'Training_R2': r2_score(y_train, y_pred_train),
            'Testing_R2': r2_score(y_test, y_pred_test),
            'Training_RMSE': calculate_rmse(y_train, y_pred_train),
            'Testing_RMSE': calculate_rmse(y_test, y_pred_test),
            'Training_MAE': mean_absolute_error(y_train, y_pred_train),
            'Testing_MAE': mean_absolute_error(y_test, y_pred_test),
            'Training_MAPE': calculate_mape(y_train, y_pred_train),
            'Testing_MAPE': calculate_mape(y_test, y_pred_test),
            'Architecture': model_name.replace('_', ' '),
            'Framework': 'Keras/TensorFlow' if 'Keras' in model_name else 'PyTorch'
        }
    except Exception as e:
        print(f"   ❌ Evaluation failed: {e}")
        return {
            'Model_Name': model_name,
            'Model_Type': model_name.replace('_', ' '),
            'Training_R2': None,
            'Testing_R2': None,
            'Training_RMSE': None,
            'Testing_RMSE': None,
            'Training_MAE': None,
            'Testing_MAE': None,
            'Training_MAPE': None,
            'Testing_MAPE': None,
            'Architecture': model_name.replace('_', ' '),
            'Framework': 'Keras/TensorFlow' if 'Keras' in model_name else 'PyTorch',
            'Error': str(e)
        }

def run_minimal_test(X_train, y_train):
    """Run a minimal test to verify TensorFlow/Keras functionality"""
    print("🧪 Running minimal TensorFlow test...")
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)
        print("   ✅ Minimal test passed")
        return True
    except Exception as e:
        print(f"   ❌ Minimal test failed: {e}")
        return False

# =====================
# Main Execution
# =====================

def main():
    """Main execution function"""
    print("🚀 DEEP LEARNING AIR POLLUTION ANALYSIS")
    print("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # Load and preprocess data
        X, y = load_and_preprocess_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        input_dim = X_train_scaled.shape[1]
        print(f"📏 Input dimension: {input_dim}")
        print(f"🎯 Training samples: {len(X_train_scaled)}, Test samples: {len(X_test_scaled)}")
        
        # Run minimal test
        if not run_minimal_test(X_train_scaled, y_train):
            print("❌ Minimal test failed. Exiting.")
            return
        
        # Define models
        models = {
            'Keras_DNN': KerasRegressor(
                model=create_dnn_model,
                model__input_dim=input_dim,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0
            ),
            'Keras_Deep_DNN': KerasRegressor(
                model=create_deep_dnn_model,
                model__input_dim=input_dim,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0
            ),
            'Keras_LSTM': SequenceModelWrapper(
                create_lstm_model,
                input_dim=input_dim,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0
            ),
            'Keras_GRU': SequenceModelWrapper(
                create_gru_model,
                input_dim=input_dim,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0
            ),
            'Keras_BiLSTM': SequenceModelWrapper(
                create_bilstm_model,
                input_dim=input_dim,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0
            ),
            'Keras_Conv1D': SequenceModelWrapper(
                create_conv1d_model,
                input_dim=input_dim,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0
            ),
            'Keras_Attention_LSTM': SequenceModelWrapper(
                create_attention_lstm_model,
                input_dim=input_dim,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0
            ),
            'Keras_Residual_DNN': KerasRegressor(
                model=create_residual_dnn_model,
                model__input_dim=input_dim,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0
            ),
            'Keras_Autoencoder': KerasRegressor(
                model=create_autoencoder_model,
                model__input_dim=input_dim,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0
            ),
            'Torch_MLP': TorchMLPRegressor(
                input_dim=input_dim,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                lr=0.001,
                verbose=0
            ),
            'Torch_ResNet': TorchResNetRegressor(
                input_dim=input_dim,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                lr=0.001,
                verbose=0
            )
        }
        
        # Train and evaluate models
        results = []
        successful_models = 0
        
        print(f"\n🎯 Training {len(models)} deep learning models...")
        print("-" * 60)
        
        for i, (model_name, model) in enumerate(models.items(), 1):
            print(f"[{i}/{len(models)}] 🔄 Training {model_name}...")
            
            try:
                # Handle Wide & Deep model separately (requires dual input)
                if model_name == 'Keras_Wide_Deep':
                    # Skip Wide & Deep for now due to complexity
                    print("   ⏭️  Skipping Wide & Deep (dual input complexity)")
                    continue
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                metrics = evaluate_model(
                    model, X_train_scaled, X_test_scaled, 
                    y_train, y_test, model_name
                )
                
                results.append(metrics)
                successful_models += 1
                
                print(f"   ✅ {model_name} completed - Test R²: {metrics['Testing_R2']:.4f}")
                
            except Exception as e:
                print(f"   ❌ {model_name} failed: {str(e)[:100]}...")
                # Add failed result
                failed_metrics = evaluate_model(
                    None, None, None, None, None, model_name
                )
                failed_metrics['Error'] = str(e)[:200]
                results.append(failed_metrics)
        
        # Save results
        if results:
            results_df = pd.DataFrame(results)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
            
            # Save to CSV
            results_df.to_csv(RESULTS_CSV, index=False)
            
            # Display summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print("\n" + "=" * 60)
            print("📊 TRAINING SUMMARY")
            print("=" * 60)
            print(f"✅ Successfully trained: {successful_models}/{len(models)} models")
            print(f"⏱️  Total training time: {duration:.1f} seconds")
            print(f"💾 Results saved to: {RESULTS_CSV}")
            
            # Show top performing models
            successful_results = results_df[results_df['Testing_R2'].notna()]
            if not successful_results.empty:
                top_models = successful_results.nlargest(3, 'Testing_R2')
                print(f"\n🏆 TOP 3 MODELS BY TEST R²:")
                for idx, row in top_models.iterrows():
                    print(f"   {row['Model_Name']}: R² = {row['Testing_R2']:.4f}")
            
            print(f"\n📈 View detailed results: {RESULTS_CSV}")
            
        else:
            print("❌ No results to save - all models failed")
            
    except Exception as e:
        print(f"❌ Critical error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 