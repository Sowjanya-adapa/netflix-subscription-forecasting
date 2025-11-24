# 📘 **Netflix Subscription Forecasting — SARIMA Time Series Project**

📈 **Forecasting future Netflix subscribers using SARIMA, seasonal decomposition, and time-series analysis.**
This project demonstrates a complete end-to-end forecasting pipeline using Python and Statsmodels.

---

## 🚀 **Project Overview**

This project focuses on modeling and forecasting quarterly Netflix subscription growth.
It uses advanced time-series techniques including:

* 📅 Datetime indexing (Year + Quarter → Time Period)
* 🔄 Resampling using quarterly frequency
* 📉 Seasonal decomposition (trend, seasonality, residual)
* 🤖 SARIMA modelling with grid search for best parameters
* 🧪 Diagnostics and forecast plots
* 🔮 Forecasting future 8 quarters (2 years)

The project provides a realistic demonstration of **business forecasting in entertainment analytics**.

---

## 📁 **Folder Structure**

```
NETFLIX-SUBSCRIPTIONS-FORECASTING/
│
├── data/
│   └── Netflix-Subscriptions.csv
│
├── src/
│   └── netflix_subscription.py
│
├── plots/          (optional; you can save output graphs here)
│
├── models/         (optional; for saving pickled models)
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📊 **Dataset Description**

`Netflix-Subscriptions.csv` includes:

| Column                      | Description               |
| --------------------------- | ------------------------- |
| **Year**                    | Year of observation       |
| **Quarter**                 | Q1, Q2, Q3, Q4            |
| **Subscribers_Millions**    | Total Netflix subscribers |
| **Revenue_Billion**         | Revenue generated         |
| **Marketing_Spend_Million** | Marketing budget          |
| **Subscription_Price**      | Netflix subscription cost |

You manually built a **Time Period** datetime index using Year + Quarter.

---

## 🧠 **Tech Stack**

### **Languages & Tools**

* Python 3.x
* Jupyter / VS Code
* Git, GitHub

### **Libraries**

* `pandas`
* `numpy`
* `matplotlib`
* `statsmodels`
* `scipy`

---

## 🔧 **How to Run the Project**

### **1️⃣ Clone the Repo**

```
git clone https://github.com/Sowjanya-adapa/netflix-subscription-forecasting.git
cd netflix-subscription-forecasting
```

### **2️⃣ Install Dependencies**

```
pip install -r requirements.txt
```

### **3️⃣ Run the Project**

```
python src/netflix_subscription.py
```

---

## 📉 **Time Series Steps Explained**

### **✔ Data Preparation**

* Convert Year + Quarter → datetime
* Sort and index using Time Period
* Convert numeric columns properly
* Resample to quarterly (QE) frequency

### **✔ Seasonal Decomposition**

* Trend → direction of subscriber growth
* Seasonality → quarterly patterns
* Residuals → noise / unexplained variation

### **✔ SARIMA Modeling**

* Grid search over `(p, d, q)` values
* Seasonal order `(P, D, Q, 4)` for quarterly data
* Best model chosen using AIC score
* Diagnostic checks for model quality

### **✔ Forecasting**

* Predict next **8 quarters**
* Plot observed vs. forecast
* Show 95% confidence intervals

---

## 📈 **Output Visuals**

This project generates:

* Original time-series plot
* Trend component
* Seasonal component
* Residual component
* SARIMA diagnostics
* Forecast with confidence interval

You can save the images manually to `/plots/`.

---

## 🧪 Example Forecast Plot

*(Insert your output image here once generated)*

```
plots/forecast.png
```

---

## 🔮 **Future Enhancements**

* Add Facebook Prophet model
* Add LSTM Neural Networks
* Add Streamlit dashboard
* Add automated model selection
* Deploy as web API
* Enable interactive visualizations

---

## 📝 **Author**

**Sowjanya Adapa**
AI & ML Undergraduate | Learning Agentic AI & Prompt Engineering
GitHub: [https://github.com/Sowjanya-adapa](https://github.com/Sowjanya-adapa)

---

## 📄 **License**
This project is licensed under the **MIT License**.

