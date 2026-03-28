# 💻 Laptop Price Predictor

An end-to-end Machine Learning project that predicts laptop prices based on various hardware and software specifications. The project includes data preprocessing, feature engineering, model building, and deployment-ready artifacts.

---

## 🚀 Project Overview

This project builds a regression model to estimate laptop prices using features such as RAM, CPU, GPU, storage, and display characteristics.

The workflow includes:
- Data cleaning and preprocessing
- Feature engineering (PPI, CPU/GPU extraction)
- Model training and evaluation
- Pipeline creation for deployment
- Model serialization using Pickle

---

## 📂 Dataset

The dataset contains laptop specifications with the following features:

- Company  
- TypeName  
- RAM  
- Weight  
- Touchscreen  
- IPS Display  
- Screen Resolution  
- CPU  
- GPU  
- HDD & SSD  
- Operating System  
- Price  

---

## ⚙️ Tech Stack

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Pickle  

---

## 🔍 Feature Engineering

Key transformations performed:

- Converted RAM from string to integer  
- Converted Weight to float  
- Extracted:
  - **CPU brand**
  - **GPU brand**
- Created new feature:
  - **PPI (Pixels Per Inch)** from resolution & screen size  
- Split storage into:
  - HDD
  - SSD  
- Encoded categorical variables using preprocessing pipeline  

---

## 🤖 Models Used

The following regression models were trained and compared:

- Linear Regression  
- Decision Tree  
- Random Forest  
- **XGBoost (High Performance Boosting Model 🚀)**  

👉 XGBoost improved performance due to:
- Handling non-linear relationships effectively  
- Built-in regularization  
- Better generalization compared to basic models  

---

## 📊 Model Evaluation

Evaluation metrics used:

- R² Score  
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  

The final model was selected based on best performance across these metrics.

---

## 💾 Model Saving

The trained pipeline and dataset are saved using pickle:

```python
pickle.dump(pipe, open('pipe.pkl','wb'))
pickle.dump(df, open('df.pkl','wb'))
