# House Price Prediction Using Regression Pipelines

This repository provides an end-to-end Machine Learning pipeline using `scikit-learn` to predict house prices based on various structural and locational features. It evaluates and compares a **Linear Regression** model against a **Polynomial Regression** model.

---

## 📊 Dataset Overview
The project processes a housing dataset (`Housing.csv`) containing 545 entries with zero missing values. The target variable is `price`.

### Features Used:
*   **Numeric Features:** `area`, `bedrooms`, `bathrooms`, `stories`, `parking`
*   **Categorical Features:** `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, `furnishingstatus`

---

## ⚙️ Project Architecture
To guarantee modularity and avoid data leakage, data preprocessing and model execution are integrated via `sklearn.pipeline.Pipeline`:

1.  **Data Partitioning:** 80% Training set and 20% Test set (`random_state=42`).
2.  **Linear Pipeline:** 
    *   Scales numeric values via `StandardScaler`.
    *   Encodes text columns via `OneHotEncoder(handle_unknown="ignore")`.
    *   Trains a standard `LinearRegression` model.
3.  **Polynomial Pipeline:** 
    *   Imputes missing values with a median strategy (via `SimpleImputer`).
    *   Generates degree-2 interactions via `PolynomialFeatures(degree=2)`.
    *   Scales data and trains a `LinearRegression` model.

---

## 📈 Model Performance Evaluation
The evaluation metrics computed on the test split are as follows:

| Model Pipeline | R² Score | RMSE | MAE |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | **0.6529** | **1,324,506.96** | **970,043.40** |
| **Polynomial Regression (Degree 2)** | 0.6496 | 1,330,810.07 | 978,314.39 |

### Key Insight:
For this specific dataset, **Linear Regression yields a slightly higher R² score and lower error margins (RMSE/MAE)**. Introducing polynomial combinations caused minor overfitting, making the simpler linear approach the optimal choice.

---

## 🚀 How to Run & Predict

### Prerequisites
Install the required packages using pip:
```bash
pip install pandas numpy scikit-learn
```

### Run Prediction Inference
The pipeline automatically routes inference data through the best performing model:

```python
# Call the built-in function to predict values
predicted_price = predict_house_price(
    area=7000, bedrooms=3, bathrooms=2, stories=2, parking=1,
    mainroad="yes", guestroom="no", basement="yes", hotwaterheating="no",
    airconditioning="yes", prefarea="yes", furnishingstatus="semi-furnished"
)

print(f"Predicted House Price: \${predicted_price:,.2f}")
# Output: Predicted House Price: \$7,423,441.38
```
# Assignment-House_Price_Regression
