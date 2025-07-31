
# 💡 SmartCustomerSegmentation

A Flask-based web application that segments banking customers into distinct clusters using K-Means clustering. This machine learning solution helps banks and financial institutions better understand customer behavior and strategize services accordingly.

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Flask](https://img.shields.io/badge/Flask-WebApp-lightgrey)
![ML](https://img.shields.io/badge/MachineLearning-KMeans-orange)
![License](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 🔍 Project Overview

Understanding customer behavior is crucial in the banking industry. This project leverages **K-Means Clustering** to segment customers based on their financial behavior (like credit usage, payments, balances, etc.). The model is integrated into a **Flask web app** that allows users to upload a `.csv` file and view real-time clustering results in table format.

---

## 🚀 Features

- 📂 Upload your own CSV with customer data
- 🧠 Automatically segments customers into meaningful clusters
- 📊 Displays result in a user-friendly HTML table
- ⚙️ Built with Flask, Pandas, and Scikit-learn
- 🎯 Model trained on real-world financial data

---

## 📁 File Structure

```
SmartCustomerSegmentation/
├── app.py                      # Flask backend
├── model.pkl                   # Trained KMeans model
├── scaler.pkl                  # Trained StandardScaler
├── CC GENERAL.csv              # Sample dataset
├── customer_segmentation.ipynb # Jupyter notebook (model training)
├── requirements.txt            # All dependencies
├── templates/
│   └── index.html              # Frontend UI
├── static/
│   └── style.css               # Optional styling
```

---

## 🧠 Machine Learning Approach

- **Model Used**: K-Means Clustering (unsupervised learning)
- **Data**: `CC GENERAL.csv` (banking customer behavior)
- **Preprocessing**:
  - Dropped non-numeric `CUST_ID`
  - Filled missing values
  - Applied `StandardScaler`
- **Cluster Optimization**:
  - Elbow Method
  - Silhouette Score

---

## 🖥️ How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/SmartCustomerSegmentation.git
cd SmartCustomerSegmentation
```

### 2. Create Virtual Environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# OR
source venv/bin/activate     # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train and Save Model (only once)

Run the last cell in `customer_segmentation.ipynb` to generate `model.pkl` and `scaler.pkl`.

### 5. Launch the Web App

```bash
python app.py
```

Then open your browser at 👉 `http://127.0.0.1:5000`

---

## 📸 Screenshots

<details>
<summary>Click to Expand</summary>

![screenshot-1](https://via.placeholder.com/800x400.png?text=Web+App+Interface)
![screenshot-2](https://via.placeholder.com/800x400.png?text=Upload+CSV+and+View+Clusters)

</details>

---

## 📦 Technologies Used

- **Python 3.9**
- **Flask**
- **Pandas, NumPy**
- **Scikit-learn**
- **HTML + CSS**

---

## ✨ Future Enhancements (Optional Ideas)

- 📈 Add PCA-based cluster visualizations
- 💾 Store uploads and cluster results in a database
- 🌐 Deploy the app using Render / Heroku / Streamlit Cloud
- 🧩 Add customer persona labeling (High Spenders, Dormant, etc.)

---

## 🙋‍♀️ Author

👩‍💻 **Yasvi Kachhadiya**  
Computer Engineering Student | Passionate about ML + Data  
📧 yasvi@example.com

---

## 🌟 Acknowledgments

- Dataset: UCI Machine Learning Repository
- Inspired by real-world banking segmentation challenges

---

## 📜 License

This project is licensed for educational and portfolio use.
