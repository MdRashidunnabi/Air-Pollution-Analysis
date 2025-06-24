# 🌍 Air Pollution Analysis & Prediction

**Comprehensive Machine Learning Analysis for Air Pollution Prediction in Ireland**

This project provides a complete machine learning pipeline for analyzing and predicting air pollution levels (NO2 concentrations) using meteorological and temporal data from Ireland.

## 📊 Project Overview

This comprehensive analysis includes:
- **17 different machine learning models** (Linear Regression, Random Forest, XGBoost, CatBoost, LightGBM, Extra Trees, etc.)
- **Complete data preprocessing** following scientific methodology
- **Advanced visualizations** (11 different figure types with 20+ plots)
- **Experimental tables** for research documentation
- **Interactive model selection** for custom visualizations
- **Automated best model detection** and evaluation

## 🎯 Key Features

### ✨ All-in-One Solution
- **Model Training**: 17 models including ensemble methods
- **Visualization Engine**: 11 comprehensive figure types
- **Table Generation**: Automated experimental documentation
- **Interactive Interface**: Custom model visualization selection

### 📈 Generated Outputs
- **Training Results**: All models saved with performance metrics
- **Visualizations**: 20+ publication-ready figures
- **Tables**: Dataset statistics, outliers analysis, model performance
- **Best Model**: Automatically identified and highlighted

### 🔧 Technical Highlights
- Proper data preprocessing with outlier detection
- Feature engineering (temporal, seasonal patterns)
- Cross-validation and performance evaluation
- Publication-ready visualizations
- No grids, bold fonts for clarity

## 📂 Project Structure

```
Air-Pollution-Analysis/
├── comprehensive_air_pollution_analysis.py  # Main analysis script
├── air_pollution_ireland.csv               # Dataset
├── requirements.txt                         # Dependencies
├── README.md                               # This file
├── trained_models/                         # Generated models (after running)
├── visualization_results/                  # Generated figures (after running)
└── experimental_results_tables/           # Generated tables (after running)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/MdRashidunnabi/Air-Pollution-Analysis.git
cd Air-Pollution-Analysis
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the analysis:**
```bash
python comprehensive_air_pollution_analysis.py
```

## 📋 Usage

### Automatic Mode
The script runs automatically and generates:
1. **All 17 models** trained and evaluated
2. **Best model identification** (highest R² score)
3. **Complete visualizations** for the best model
4. **Experimental tables** in CSV format

### Interactive Mode
After automatic analysis, you can:
- Select any model for custom visualizations
- Generate specific figure sets
- Compare different model performances

### Example Run:
```bash
$ python comprehensive_air_pollution_analysis.py

============================================================
ULTIMATE COMPREHENSIVE AIR POLLUTION ANALYSIS
Training Models + Tables + Interactive Visualizations
============================================================

🏆 Best model: Extra_Trees (R² = 0.7281)
✅ All models trained and saved in: trained_models/
✅ Best model visualizations: visualization_results/best_model_extra_trees/

Do you want to see visualizations for any other model?
Enter 'yes' to continue or 'no' to exit: yes
```

## 📊 Dataset

**Source**: Air pollution monitoring data from Ireland
- **Samples**: 56,757 records (after preprocessing)
- **Features**: 13 meteorological and temporal variables
- **Target**: NO2 concentration (µg/m³)
- **Time Period**: Multi-year environmental monitoring data

### Features Included:
- **Temporal**: Year, Month, Day, Hour, Season
- **Meteorological**: Temperature, Humidity, Pressure, Wind Speed/Direction
- **Environmental**: Dew point, Cloud amount, Industrial index

## 🎨 Visualizations Generated

The script generates **11 comprehensive figure types**:

### 📈 Core Analysis Figures
1. **Figure 2**: Negative NO2 Analysis
2. **Figure 3**: Feature Importance Analysis
3. **Figure 4**: Environmental Parameter Histograms
4. **Figure 5**: Correlation Heatmap
5. **Figure 6**: PCA Loading Analysis

### 🔍 Advanced Analysis
6. **Figure 7**: NO2 vs Variables Scatterplots
7. **Figure 8**: Actual vs Predicted Values
8. **Figure 9**: Temporal Patterns (7 sub-figures)
9. **Figure 10**: Wind Direction Circular Analysis
10. **Figure 11**: Seasonal Patterns (7 sub-figures)

**Total**: 20+ individual plots generated automatically!

## 🤖 Machine Learning Models

### Base Models (14):
- Linear Regression
- Random Forest
- Support Vector Regressor
- Multi-layer Perceptron
- XGBoost
- Decision Tree
- K-Nearest Neighbors
- AdaBoost
- Gradient Boosting
- Lasso
- Ridge
- ElasticNet
- CatBoost
- LightGBM

### Ensemble Models (3):
- Extra Trees Regressor
- Bagging with Random Forest
- Voting Regressor

## 📈 Performance Metrics

All models evaluated using:
- **R² Score** (coefficient of determination)
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **Training vs Test** performance comparison

## 📄 Output Files

### Generated Directories:
```
trained_models/                    # All 17 trained models (.pkl files)
├── best_model_info.txt
├── scaler.pkl
└── [model_name]_model.pkl

visualization_results/              # Publication-ready figures
├── best_model_[model_name]/       # Best model visualizations
└── [other_models]/                # Additional model visualizations

experimental_results_tables/       # Research documentation
├── Table1_Dataset_Statistics.csv
├── Table2_Outliers_Detected.csv
└── Table4_Model_Performance_Metrics.csv
```

## ⚙️ Configuration

### Key Parameters (modifiable in script):
```python
DATASET_FILE = 'air_pollution_ireland.csv'  # Dataset filename
RANDOM_STATE = 42                           # Reproducibility seed
```

### Customization Options:
- Model hyperparameters
- Visualization styles
- Output directories
- Feature selection

## 🧪 Research Applications

Perfect for:
- **Environmental Science Research**
- **Machine Learning Comparison Studies**
- **Air Quality Prediction Models**
- **Publication-Ready Visualizations**
- **Educational Demonstrations**

## 🏆 Results Summary

**Best Model Performance** (typical results):
- **Model**: Extra Trees Regressor
- **R² Score**: ~0.73 (73% variance explained)
- **RMSE**: ~9.8 µg/m³
- **Features**: Temperature, humidity, and temporal patterns most important

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit changes (`git commit -am 'Add improvement'`)
5. Push to branch (`git push origin feature/improvement`)
6. Create a Pull Request



## 👨‍💻 Author

**Md Rashidunnabi**
- GitHub: [@MdRashidunnabi](https://github.com/MdRashidunnabi)
- Repository: [Air-Pollution-Analysis](https://github.com/MdRashidunnabi/Air-Pollution-Analysis.git)

## 📧 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/MdRashidunnabi/Air-Pollution-Analysis/issues) page
2. Create a new issue with detailed description
3. Include error messages and system information

## 🙏 Acknowledgments

- Environmental monitoring agencies for data collection
- Open source machine learning community
- Scikit-learn, XGBoost, CatBoost, and LightGBM development teams

---

**⭐ Star this repository if you find it useful!**

*This project demonstrates comprehensive machine learning analysis for environmental science applications.* 