🧬 FitPulse — Unified Health Analytics App

💪 AI-powered health analytics pipeline for fitness data
🚀 Built using Streamlit + Machine Learning + Time Series Analysis

📌 Overview

FitPulse is an end-to-end health analytics application that processes fitness tracker data (Fitbit dataset) and transforms it into meaningful insights using:

🧹 Data Cleaning
🧪 Feature Engineering (TSFresh)
📈 Time Series Forecasting (Prophet)
🤖 Clustering (KMeans + DBSCAN)
📊 Dimensionality Reduction (PCA + t-SNE)

It helps in detecting anomalies, understanding user behavior, and predicting future trends.

✨ Features
🧹 Data Cleaning Module
Upload CSV files
Automatic null handling:
Numeric → Forward/Backward Fill
Categorical → "No Workout"
Datetime parsing
Null value comparison (Before vs After)
🧬 ML Pipeline
🔧 Data Loading & Parsing
Combines 5 Fitbit datasets:
Daily Activity
Hourly Steps
Hourly Intensities
Sleep Data
Heart Rate Data
Builds a unified master dataset
🧪 TSFresh Feature Extraction
Extracts statistical features from heart rate time-series
Normalizes features
Heatmap visualization for feature importance
📈 Prophet Forecasting
Predicts:
❤️ Heart Rate
👣 Steps
💤 Sleep
30-day future forecasting
Confidence intervals (80%)
🤖 Clustering & Segmentation
KMeans → User segmentation
DBSCAN → Noise & anomaly detection
PCA + t-SNE → Visualization
🧑‍🤝‍🧑 User Personas
Sedentary 🛋️
Moderately Active 🚶
Highly Active 🏃
🛠️ Tech Stack
Frontend: Streamlit
Data Processing: Pandas, NumPy
Visualization: Matplotlib, Seaborn, Plotly

ML Algorithms:

TSFresh
Prophet
Scikit-learn (KMeans, DBSCAN, PCA, t-SNE)
📂 Dataset
📊 Fitbit Dataset (March–April 2016)
~30 users
Minute-level heart rate data
🚀 How to Use
🧹 Upload CSV file in Data Cleaning section
🧬 Upload all 5 Fitbit datasets
🔧 Click Load & Parse
🧪 Run TSFresh Feature Extraction
📈 Run Forecasting
🤖 Run Clustering
📊 Output
Cleaned dataset preview
Feature heatmaps
Forecast graphs
Cluster visualizations (PCA & t-SNE)
User personas
🎯 Use Cases
Health monitoring systems
Fitness analytics dashboards
Wearable device data analysis
Anomaly detection in health data
📌 Future Improvements
🔮 Real-time data integration
🤖 Deep Learning models (LSTM)
📱 Mobile-friendly UI
☁️ Deployment on cloud (AWS / Streamlit Cloud)
👨‍💻 Author

Vijay Bawaskar
🎓 Computer Science Engineer (Data Science)
💡 Interested in ML, Data Science & AI
