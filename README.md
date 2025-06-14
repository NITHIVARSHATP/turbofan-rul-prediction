# Turbofan Engine Remaining Useful Life (RUL) Prediction

## Project Overview

This project focuses on predicting the Remaining Useful Life (RUL) of turbofan engines based on multivariate time series sensor data. Accurate RUL estimation is crucial for proactive maintenance, reducing downtime, and preventing catastrophic failures in aerospace engines.

The dataset used is a simulation of engine degradation over time with sensor measurements capturing various operating conditions. The goal is to develop machine learning models that can predict how many operational cycles remain before engine failure.

---

## Folder Structure

turbofan-rul-prediction/
├── data/ # Raw and processed datasets
├── notebooks/ # Jupyter notebooks for exploration, EDA, and modeling
├── models/ # Saved trained models and checkpoints
├── reports/ # Reports, visualizations, and presentations
├── .gitignore # Git ignore file
├── README.md # Project documentation
├── requirements.txt # Python dependencies list



---

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Create and Activate Virtual Environment (Windows)

Open Command Prompt or PowerShell and run:

```
cmd
cd "path\to\turbofan-rul-prediction"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

pip install numpy pandas scikit-learn matplotlib seaborn tensorflow jupyter
```
Dataset
The project uses the NASA C-MAPSS turbofan engine degradation simulation dataset.

You can download it from:
https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/

Place the downloaded dataset files inside the data/ directory.

How to Use
Activate your virtual environment (see setup instructions).

Launch Jupyter Notebook:
jupyter notebook
Open the notebooks inside the notebooks/ folder to explore the data, preprocess, and train models.

Trained models will be saved inside the models/ directory.

Reports and visualizations can be found in the reports/ folder.

Key Technologies & Libraries
Python 3.8+

NumPy & Pandas for data manipulation

Scikit-learn for machine learning algorithms and evaluation

TensorFlow / Keras for deep learning models

Matplotlib & Seaborn for data visualization

Jupyter Notebook for interactive development

Project Roadmap
Data Exploration & Visualization

Data Cleaning and Feature Engineering

Baseline Model Development (Regression, Random Forest, etc.)

Advanced Modeling with Deep Learning (LSTM, CNN)

Model Evaluation & Hyperparameter Tuning

Final Analysis and Reporting

Author
Nithivarsha T P

License
This project is licensed under the MIT License. See the LICENSE file for details.
# turbofan-rul-prediction
Predict Remaining Useful Life (RUL) of turbofan engines using NASA's C-MAPSS dataset with machine learning and deep learning (LSTM) models.
