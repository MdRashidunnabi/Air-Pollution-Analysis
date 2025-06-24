#!/usr/bin/env python3
"""
Setup script for Air Pollution Analysis project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="air-pollution-analysis",
    version="1.0.0",
    author="Md Rashidunnabi",
    author_email="your.email@example.com",
    description="Comprehensive Machine Learning Analysis for Air Pollution Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MdRashidunnabi/Air-Pollution-Analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "air-pollution-analysis=comprehensive_air_pollution_analysis:main",
        ],
    },
) 