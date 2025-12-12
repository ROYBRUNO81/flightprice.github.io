# ✈️ Flight Fare Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **A comprehensive machine learning analysis predicting flight fares using the Bangladesh Flight Price Dataset (57,000+ records)**

[View Full Analysis](https://github.com/ROYBRUNO81/flightprice.github.io/blob/main/Project.ipynb) | [Dataset Source](https://www.kaggle.com/datasets/mahatiratusher/flight-price-dataset-of-bangladesh)

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Technical Approach](#technical-approach)
- [Results](#results)
- [Installation & Usage](#installation--usage)
- [Technologies Used](#technologies-used)
- [Contributors](#contributors)

---

## 🎯 Project Overview

This project explores the **key drivers of flight fare variability** within the Bangladesh aviation ecosystem through comprehensive data analysis and machine learning. We built predictive models to forecast ticket prices based on 17 distinct features including airline operations, route characteristics, booking timing, and seasonal patterns.

### **Business Problem**
Understanding what influences airline ticket pricing is crucial for both travelers seeking the best deals and airlines optimizing their revenue strategies. This analysis identifies which factors, from cabin class to departure timing, have the most significant impact on fare costs.

### **Objectives**
- Predict total flight fare (BDT) with high accuracy
- Identify the most influential pricing factors
- Compare linear vs. non-linear modeling approaches
- Provide actionable insights for travelers and industry stakeholders

---

## ✨ Key Features

### **Data Processing & Engineering**
- ✅ Cleaned and validated 57,000+ flight records
- ✅ Engineered 15+ predictive features including:
  - Temporal patterns (month, day, hour, weekend indicators)
  - Route frequency metrics
  - Numeric stopover conversions
  - Calendar-based seasonality signals

### **Exploratory Data Analysis**
- 📊 Comprehensive visualization suite examining:
  - Price distributions and outlier patterns
  - Seasonal fare trends (Regular, Eid, Hajj, Winter Holidays)
  - Class-based pricing tiers (Economy → Business → First)
  - Route-specific characteristics
  - Correlation analysis across numerical features

### **Machine Learning Pipeline**
- 🔧 Production-ready sklearn pipeline with:
  - Automated preprocessing (imputation, scaling, encoding)
  - Multiple model architectures tested
  - Robust train-test validation
  - Hyperparameter optimization

---

## 📊 Dataset

**Source:** Flight Price Dataset of Bangladesh  
**Size:** 57,000 simulated flight records  
**Features:** 17 original columns including:

| Feature | Description |
|---------|-------------|
| **Airline** | Carrier operating the flight |
| **Aircraft Type** | Model of aircraft |
| **Source/Destination** | Airport codes for departure/arrival |
| **Class** | Economy, Business, or First Class |
| **Duration** | Flight time in hours |
| **Stopovers** | Direct, 1 Stop, or 2 Stops |
| **Days Before Departure** | Booking advance window (1-90 days) |
| **Seasonality** | Regular, Eid, Hajj, or Winter Holidays |
| **Total Fare (BDT)** | **Target variable** |

*Note: Base Fare and Tax & Surcharge were excluded from modeling to prevent data leakage*

---

## 🔬 Technical Approach

### **1. Data Preprocessing**
```python
# Key preprocessing steps
- Removed redundant columns (Source Name, Destination Name)
- Converted datetime strings → numeric features (month, hour, day of week)
- Mapped categorical stopovers → numeric values (Direct=0, 1 Stop=1, 2 Stops=2)
- Created route frequency feature (flight popularity metric)
- One-hot encoded categorical variables
- Standardized numeric features
```

### **2. Feature Engineering Highlights**
- **Temporal Features:** Extracted `dep_month`, `dep_hour`, `dep_dayofweek`, `dep_is_weekend`
- **Route Analysis:** Computed `route_frequency` to capture demand patterns
- **Stopover Encoding:** Converted text labels to numeric progression
- **Leakage Prevention:** Dropped Base Fare and Tax columns (sum to target)

### **3. Modeling Strategy**

We evaluated **6 model configurations** across two target representations:

#### **Linear Baselines**
| Model | Target | R² | RMSE (BDT) | Notes |
|-------|--------|-----|------------|-------|
| Linear Regression | Raw | 0.570 | 53,538 | Baseline OLS |
| Ridge (α=1) | Raw | 0.570 | 53,538 | No improvement—features well-conditioned |
| Linear Regression | log1p | 0.651 | 48,265 | **~10% RMSE reduction** |
| Ridge (α=1) | log1p | 0.651 | 48,269 | Matches log OLS performance |

**Insight:** Log-transform helped linear models; L2 regularization showed no benefit.

#### **Tree Ensembles**
| Model | Target | R² | RMSE (BDT) | Configuration |
|-------|--------|-----|------------|---------------|
| RandomForest | Raw | **0.663** | 47,400 | 200 trees, min_samples_leaf=2 |
| RandomForest | log1p | 0.638 | 49,093 | Log target underperforms |
| **HistGradientBoosting** | **Raw** | **0.677** | **46,437** | **50 iterations, lr=0.1, l2=0.5** ✅ |
| HistGradientBoosting | log1p | 0.651 | 48,226 | Log target slightly inferior |

---

## 🏆 Results

### **Best Model: HistGradientBoosting (Raw Target)**
```
✅ R² Score: 0.677 (explains 67.7% of fare variance)
✅ RMSE: 46,437 BDT (~$386 USD)
✅ 13% improvement over baseline Linear Regression
```

### **Key Findings**

1. **Class is the dominant predictor**  
   - First Class costs ~300% more than Economy
   - Business costs ~100% more than Economy
   - Clear pricing tiers with high separation

2. **Seasonality drives major price swings**  
   - Hajj period: +42% vs. Regular season
   - Eid period: +35% vs. Regular season
   - Winter Holidays: +18% vs. Regular season

3. **Duration shows weak correlation with price** (r ≈ 0.33)  
   - Suggests other factors (demand, competition) dominate pricing

4. **Airline choice has minimal impact**  
   - Only ~12% fare difference between carriers
   - Route and class matter far more

5. **Booking timing shows no clear pattern**  
   - Days before departure: r ≈ -0.07 (nearly zero correlation)
   - Challenges the "book early = cheaper" conventional wisdom

### **Model Comparison Visualization**
```
Model Performance (RMSE in BDT):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline OLS       ████████████████████ 53,538
Log OLS            ████████████████     48,265
RandomForest       ███████████████      47,400
HistGradientBoosting ██████████████     46,437 ⭐ BEST
```

---

## 🚀 Installation & Usage

### **Prerequisites**
```bash
Python 3.8+
pip install -r requirements.txt
```

### **Dependencies**
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### **Quick Start**
```bash
# Clone the repository
git clone https://github.com/yourusername/flight-fare-prediction.git
cd flight-fare-prediction

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook Flight_Fare_Prediction_Final.ipynb
```

### **Running the Analysis**
1. **Load Data:** Execute cells in Section 2.1
2. **Preprocess:** Run through Section 2.3
3. **EDA:** Explore visualizations in Section 3
4. **Feature Engineering:** Execute Section 4
5. **Model Training:** Run all model cells in final section
6. **Evaluate:** Review performance metrics and comparisons

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| **Languages** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn (LinearRegression, Ridge, RandomForest, HistGradientBoosting) |
| **Development** | Jupyter Notebook, Google Colab |
| **Version Control** | Git, GitHub |

---

## 👥 Contributors

**Ange Christa Dushime** | **Christian Ishimwe** | **Bruno Ndiba Mbwaye Roy**

*CIS 5450 Final Project — University of Pennsylvania*

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

</div>
