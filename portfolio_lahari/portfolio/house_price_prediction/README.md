# 🏠 House Price Prediction
**By Tetali Lakshmi Nagalahari**

A machine learning web application that predicts California house prices using Linear Regression, Random Forest, and Gradient Boosting models, with an interactive Streamlit dashboard.

---

## 📁 Project Structure

```
house_price_prediction/
├── app.py               ← Streamlit web app (main file)
├── train_model.py       ← Standalone training script + saves plots
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Train & save model + plots
```bash
python train_model.py
```

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```
Then open **http://localhost:8501** in your browser.

---

## 🤖 Models Used

| Model               | Description                                      |
|---------------------|--------------------------------------------------|
| Linear Regression   | Baseline model — fast, interpretable             |
| Random Forest       | Ensemble of decision trees — handles non-linearity|
| Gradient Boosting   | Boosted trees — highest accuracy                 |

---

## 📊 Features

- **11 engineered features** from the California Housing dataset
- Feature engineering: `RoomsPerHousehold`, `BedroomsPerRoom`, `PopulationPerHousehold`
- **Evaluation metrics**: R², MAE, RMSE, 5-Fold Cross-Validation
- **Interactive Streamlit dashboard** with:
  - 🔮 Real-time price prediction from sidebar sliders
  - 📊 Model comparison charts
  - 📈 Actual vs Predicted scatter plots
  - 🗺️ Residuals analysis
  - 🔍 Feature importance (RF & GB)
  - 📉 Correlation heatmap

---

## 📈 Results (approximate)

| Model              | R²     | MAE    | RMSE   |
|--------------------|--------|--------|--------|
| Linear Regression  | ~0.60  | ~0.53  | ~0.72  |
| Random Forest      | ~0.81  | ~0.33  | ~0.50  |
| Gradient Boosting  | ~0.78  | ~0.37  | ~0.53  |

---

## 🛠️ Tech Stack

- **Python** · **Pandas** · **NumPy**
- **Scikit-learn** · **Matplotlib** · **Seaborn**
- **Streamlit**

---

## 📬 Contact

**Tetali Lakshmi Nagalahari**  
📧 laharireddytetali@gmail.com  
📍 Bhimavaram, Andhra Pradesh
