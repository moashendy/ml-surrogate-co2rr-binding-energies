# Machine Learning Surrogate Model for CO₂RR Adsorption Energies

## Project Motivation

This project develops a machine learning surrogate model to predict adsorption
(binding) energies of CO₂RR intermediates on catalyst surfaces, enabling
rapid screening of materials.

## Project Overview
Density Functional Theory (DFT) calculations of CO₂ reduction reaction (CO₂RR)
intermediates are accurate but computationally expensive.  
This project develops a machine-learning surrogate model to predict adsorption energies of key CO₂ reduction intermediates (e.g., CO*, COOH*, OCHO*) on metallic and alloy catalyst surfaces, enabling rapid screening of materials.

The goal is to demonstrate a physics-aware ML workflow that bridges:
- Density Functional Theory (DFT)–inspired descriptors
- Feature engineering grounded in surface science
- Interpretable and generalizable regression models

## Dataset
The dataset consists of catalyst–adsorbate configurations with physically motivated descriptors, which are d-band center descriptor, Pauling electronegativity, atomic radius, valence electron count and adsorption_energy as a target variable

## Methods
- Exploratory Data Analysis (EDA)
- Feature engineering
- Regression models:
  - Linear Regression
  - Random Forest
  - Gradient Boosting / XGBoost
- Cross-validation and error analysis

## Key Insights
- Linear models confirm classical d-band theory trends
- Nonlinear models capture higher-order interactions
- Random Forest balances accuracy and generalization
- XGBoost shows signs of overfitting for current dataset size

## Tech Stack
- Python
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- Jupyter
- Git/GitHub

## Author
Mohamed Hendy  
