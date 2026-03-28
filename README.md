# Laptop_Prediction_model

# 💻 Laptop Price Predictor (ML Project)

An end-to-end Machine Learning project that predicts the price of a laptop based on its specifications like RAM, CPU, GPU, storage, etc.

---

## 🚀 Project Overview

This project uses regression techniques to estimate laptop prices. It includes:

- Data preprocessing & feature engineering  
- Model building using multiple algorithms  
- Performance evaluation  
- Deployment using Streamlit  

---

## 📂 Dataset

- Contains ~1300 laptop records  
- Features include:
  - Company
  - TypeName
  - RAM
  - Weight
  - Touchscreen
  - IPS Display
  - Screen Resolution
  - CPU
  - GPU
  - Storage (HDD/SSD)
  - Operating System
  - Price  

---

## ⚙️ Tech Stack

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Streamlit  
- Pickle  

---

## 🔍 Feature Engineering

- Converted RAM & Weight to numeric values  
- Extracted:
  - CPU brand  
  - GPU brand  
- Created new feature:
  - **PPI (Pixels Per Inch)**  
- Split storage into SSD & HDD  

---

## 🤖 Models Used

- Linear Regression  
- Decision Tree  
- Random Forest (Best Performance)  

---

## 📊 Model Evaluation

- R² Score  
- MAE (Mean Absolute Error)  
- RMSE  

---

## 💾 Model Saving

```python
pickle.dump(pipe, open('pipe.pkl','wb'))
pickle.dump(df, open('df.pkl','wb'))
