# 📦 E-Commerce Delivery Performance Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

> A machine learning project to predict whether e-commerce products will be delivered on time, helping optimize logistics operations and enhance customer satisfaction.


## 🎯 Project Overview

In the competitive world of e-commerce, timely delivery is crucial for customer satisfaction and business success. This project develops a predictive model to forecast delivery performance for an international e-commerce company specializing in electronic products. By analyzing customer behavior patterns and logistics factors, the model identifies critical elements affecting delivery timelines and provides actionable insights for operational optimization.

### Key Objectives
- Predict whether products will reach customers on time
- Analyze factors influencing delivery performance
- Study customer behavior patterns related to delivery
- Provide data-driven insights for logistics optimization

## 🚀 Features

- **Comprehensive Data Analysis**: In-depth exploratory data analysis of customer interactions and delivery patterns
- **Multiple ML Models**: Implementation and comparison of various classification algorithms
- **Feature Engineering**: Advanced preprocessing and transformation techniques
- **Performance Visualization**: Detailed confusion matrices and model comparison charts
- **Actionable Insights**: Identification of key factors affecting delivery timelines

## 📊 Dataset

The dataset comprises **10,999 observations** across **12 variables**, detailing customer interactions, product characteristics, and delivery outcomes.

### Data Dictionary

| Feature | Description |
|---------|-------------|
| **ID** | Unique identifier for customers |
| **Warehouse_block** | Warehouse storage block (A, B, C, D, F) |
| **Mode_of_Shipment** | Shipping method (Flight, Ship, Road) |
| **Customer_care_calls** | Number of customer service calls made |
| **Customer_rating** | Customer satisfaction rating (1-5, where 1 is worst, 5 is best) |
| **Cost_of_the_Product** | Product price in US Dollars |
| **Prior_purchases** | Number of previous purchases by the customer |
| **Product_importance** | Product categorization (Low, Medium, High) |
| **Gender** | Customer gender (Male, Female) |
| **Discount_offered** | Discount percentage offered on the product |
| **Weight_in_gms** | Product weight in grams |
| **Reached.on.Time_Y.N** | **Target Variable**: 1 = NOT reached on time, 0 = reached on time |

## 🛠️ Technologies Used

### Core Libraries
```python
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
```

### Machine Learning Models
- Random Forest Classifier
- Decision Tree Classifier
- Logistic Regression
- K-Nearest Neighbors (KNN)

## 📁 Project Structure

```
E-Commerce-Delivery-Performance-Prediction/
│
├── data/
│   └── E_Commerce.csv                          # Dataset
│
├── notebooks/
│   └── E-Commerce Delivery Performance Prediction.ipynb
│
├── models/
│   └── saved_models/                           # Trained model files
│
├── visualizations/
│   ├── correlation_heatmap.png
│   ├── confusion_matrices.png
│   └── model_comparison.png
│
├── requirements.txt                            # Project dependencies
├── README.md                                   # Project documentation
└── LICENSE                                     # License file
```

## 🔧 Installation

### Prerequisites
Ensure you have Python 3.8 or higher installed on your system.

### Step 1: Clone the Repository
```bash
git clone https://github.com/Aayushvsv/E-Commerce-Delivery-Performance-Prediction.git
cd E-Commerce-Delivery-Performance-Prediction
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Launch Jupyter Notebook
```bash
jupyter notebook
```

Navigate to the notebook file and run the cells sequentially.

## 📈 Methodology

### 1. Data Preprocessing
- **Data Cleaning**: Handled missing values, duplicates, and irrelevant columns
- **Data Type Verification**: Ensured appropriate data types for all features
- **Label Encoding**: Transformed categorical variables (Warehouse_block, Mode_of_Shipment, Product_importance, Gender)

### 2. Exploratory Data Analysis (EDA)
- Investigated distribution of variables
- Analyzed customer behavior patterns
- Examined logistics factors using comprehensive visualizations
- Created correlation matrix heatmap to identify relationships
- <img width="1214" height="755" alt="image" src="https://github.com/user-attachments/assets/cd9dff7c-5437-473f-9de2-5edfc80e5336" />


### 3. Feature Engineering
- Applied label encoding to categorical features
- Normalized numerical features where appropriate
- Created derived features for enhanced model performance
- <img width="1386" height="808" alt="image" src="https://github.com/user-attachments/assets/b4257e95-9be1-4b2d-956a-025fecd67133" />


### 4. Model Development
Implemented and trained four different machine learning models:
- **Random Forest Classifier**: Ensemble learning approach
- **Decision Tree Classifier**: Tree-based decision making
- **Logistic Regression**: Probabilistic classification
- **K-Nearest Neighbors**: Instance-based learning

### 5. Model Evaluation
Assessed models using multiple metrics:
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- Comparative visualization

## 📊 Results

### Model Performance Comparison

| Model | Accuracy |
|-------|----------|
| **Decision Tree Classifier** | **69%** 🏆 |
| Random Forest Classifier | 68% |
| Logistic Regression | 67% |
| K-Nearest Neighbors | 65% |

### Key Findings

1. **Decision Tree Classifier** demonstrated the highest accuracy at 69%, making it the best-performing model for this dataset.

2. **Cost-Performance Relationship**: Strong positive correlation between product cost and customer care calls. Customers are more concerned about expensive products and make more calls to check delivery status.

3. **Critical Insight**: Ensuring timely delivery is especially crucial for high-cost items to maintain customer satisfaction and reduce support overhead.

4. **Logistics Factors**: Warehouse block location and shipment mode significantly impact delivery performance.

## 🔍 Key Insights for Business

### Actionable Recommendations

1. **Prioritize High-Value Products**: Implement expedited handling for expensive items to reduce customer anxiety and support calls.

2. **Optimize Warehouse Operations**: Focus on improving efficiency in warehouse blocks with lower on-time delivery rates.

3. **Shipment Mode Analysis**: Evaluate and optimize shipping methods based on delivery performance data.

4. **Proactive Communication**: For high-value orders, implement automated status updates to reduce customer care calls.

5. **Weight Management**: Consider product weight in delivery time estimates for more accurate predictions.

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Improvement
- Implement deep learning models for comparison
- Add real-time prediction capability
- Create a web interface for model deployment
- Incorporate additional features (weather, traffic data)
- Implement hyperparameter tuning for better performance

## 📝 Future Enhancements

- [ ] Deploy model as REST API using Flask/FastAPI
- [ ] Build interactive dashboard for real-time monitoring
- [ ] Implement ensemble methods for improved accuracy
- [ ] Add time-series analysis for seasonal patterns
- [ ] Integrate with cloud platforms (AWS/Azure/GCP)
- [ ] Develop mobile application for predictions

