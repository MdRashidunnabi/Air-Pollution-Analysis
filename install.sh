#!/bin/bash

# Air Pollution Analysis - Installation Script
# Author: Md Rashidunnabi

echo "🌍 Air Pollution Analysis - Installation Script"
echo "================================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✅ Python found: $python_version"
else
    echo "❌ Python3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check pip
echo "Checking pip..."
pip_version=$(pip3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✅ Pip found: $pip_version"
else
    echo "❌ Pip not found. Please install pip."
    exit 1
fi

# Create virtual environment (optional)
echo ""
read -p "Do you want to create a virtual environment? (y/n): " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv air_pollution_env
    source air_pollution_env/bin/activate
    echo "✅ Virtual environment created and activated"
fi

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

if [[ $? -eq 0 ]]; then
    echo "✅ All requirements installed successfully!"
else
    echo "❌ Failed to install requirements. Please check the error messages above."
    exit 1
fi

# Check dataset
echo ""
if [[ -f "air_pollution_ireland.csv" ]]; then
    echo "✅ Dataset found: air_pollution_ireland.csv"
else
    echo "⚠️  Warning: Dataset 'air_pollution_ireland.csv' not found."
    echo "   Please ensure the dataset is in the project directory."
fi

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "To run the analysis:"
echo "  python comprehensive_air_pollution_analysis.py"
echo ""
echo "If you created a virtual environment, activate it first:"
echo "  source air_pollution_env/bin/activate"
echo ""
echo "Happy analyzing! 🚀" 