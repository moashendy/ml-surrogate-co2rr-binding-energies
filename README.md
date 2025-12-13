# Machine Learning Surrogate Model for CO₂RR Binding Energies

## Project Motivation
Density Functional Theory (DFT) calculations of CO₂ reduction reaction (CO₂RR)
intermediates are accurate but computationally expensive.  
This project develops a machine learning surrogate model to predict adsorption
(binding) energies of CO₂RR intermediates on catalyst surfaces, enabling
rapid screening of materials.

## Dataset
- Synthetic dataset inspired by DFT-computed binding energies
- Features represent elemental and surface descriptors
- Target: adsorption energy (eV)

## Methods
- Exploratory Data Analysis (EDA)
- Feature engineering
- Regression models:
  - Linear Regression
  - Random Forest
  - Gradient Boosting / XGBoost
- Cross-validation and error analysis

## Results
- ML models achieve significantly lower error than linear baselines
- Feature importance analysis provides physical interpretability

## Tech Stack
- Python
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- Jupyter
- Git/GitHub

## Author
Mohamed Hendy  
