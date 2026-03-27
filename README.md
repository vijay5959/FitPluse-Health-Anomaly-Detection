# 💪 FitPulse — AI-Powered Health Analytics App

An end-to-end **Health Analytics & Anomaly Detection System** built using **Streamlit, Machine Learning, and Time-Series Analysis**.

FitPulse processes real-world Fitbit data and transforms it into actionable insights using a complete ML pipeline.

---

## 🚀 Project Overview

FitPulse is divided into **3 key milestones**:

- 🧹 **Milestone 1: Data Cleaning**
- 🧬 **Milestone 2: ML Pipeline**
- 🚨 **Milestone 3: Anomaly Detection & Visualization**

---

## 📂 Dataset

- Fitbit Fitness Tracker Dataset  
- ~30 users  
- Data from March–April 2016  

Includes:
- Daily Activity
- Hourly Steps
- Hourly Intensities
- Minute Sleep Data
- Heart Rate (second-level)

---

## 🧹 Milestone 1 — Data Cleaning

### Features
- Upload any CSV file
- Automatic cleaning:
  - Date conversion
  - Missing value handling (ffill/bfill)
  - Categorical filling ("No Workout")
- Null value comparison (Before vs After)

### Output
- Clean dataset
- Preprocessing logs
- Null value charts

---

## 🧬 Milestone 2 — Machine Learning Pipeline

### Step 1: Data Loading
- Merges 5 datasets into a unified **Master DataFrame**
- Handles time normalization
- Aggregates heart rate data

---

### Step 2: TSFresh Feature Extraction
- Extracts statistical features from time-series data
- Uses:
  - MinimalFCParameters
  - MinMax Scaling

---

### Step 3: Forecasting (Prophet)
- Predicts:
  - Heart Rate
  - Steps
  - Sleep
- 30-day forecast
- Weekly seasonality
- Confidence intervals

---

### Step 4: Clustering
- KMeans (User segmentation)
- DBSCAN (Outlier detection)
- PCA & t-SNE (Visualization)

### Cluster Personas
- 🏃 Highly Active
- 🚶 Moderately Active
- 🛋️ Sedentary

---

## 🚨 Milestone 3 — Anomaly Detection

### Methods Used

#### 1. Threshold-Based
- Heart Rate > 100 or < 50
- Steps < 500 or > 25000
- Sleep < 60 or > 600

#### 2. Statistical Method
- Rolling Median
- Residual Detection (±2σ)

#### 3. Pattern-Based
- Sudden spikes/drops

---

### Output
- Detected anomalies
- Reason for anomaly
- Interactive visualizations

---

## 🎯 Accuracy Simulation
- Injects artificial anomalies
- Measures detection performance

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **Machine Learning:**
  - Scikit-learn
  - TSFresh
  - Prophet  

---

## ⚙️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/FitPulse.git
cd FitPulse
