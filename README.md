# 🏡 Simple House Price Prediction (Linear Regression)

> **Course:** Machine Learning `[ADPC509]`  
> **Unit:** Unit II — Supervised Learning: Regression  
> **Activity:** Activity 2 — Simple House Price Prediction  
> **Department:** Artificial Intelligence & Data Science (AI & DS)  
> **Institution:** Zeal College of Engineering and Research, Pune  

---

## 📌 Project Overview

This project implements a foundational **Supervised Learning** workflow using **Simple Linear Regression** to predict property values based on socioeconomic data. The model investigates the relationship between the median income of a district block and its corresponding median house price using the benchmark **California Housing Dataset**.

---

## 📊 Dataset Details

| Attribute | Details |
| :--- | :--- |
| **Dataset Source** | California Housing Dataset (`sklearn.datasets.fetch_california_housing`) |
| **Records & Features** | 20,640 instances, 8 numerical features |
| **Input Feature ($X$)** | `MedInc` (Median income in block group, expressed in tens of thousands of USD: $\$10,000$s) |
| **Target Variable ($y$)** | `MedHouseVal` (Median house value, expressed in hundreds of thousands of USD: $\$100,000$s) |
| **Missing Values** | 0 missing values across all columns |

### Feature Relevance
Economically, household income is the primary constraint and determinant of real estate purchasing power. Areas with higher median incomes command significantly higher median property values due to demand elasticity and regional desirability.

---

## ⚙️ Methodology & Step-by-Step Workflow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Load & Inspect │ ──> │   Feature /     │ ──> │ 80:20 Train-Test│
│  Dataset        │     │  Target Split   │     │      Split      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ New Observation │ <── │  Model Metrics  │ <── │ Linear Model    │
│   Prediction    │     │  & Visuals      │     │    Fitting      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

1. **Data Inspection & Cleaning:** Checked column data types, missing records, and duplicate rows.
2. **Exploratory Data Analysis (EDA):** Plotted a scatter plot ($X$ vs. $y$) revealing a clear positive upward relationship.
3. **Train-Test Split:** Separated the dataset into an $80:20$ ratio (`random_state=42`) ensuring unseen evaluation data.
4. **Model Training:** Fitted a `LinearRegression` model using ordinary least squares (OLS).
5. **Evaluation:** Assessed generalization performance using Mean Absolute Error ($MAE$), Mean Squared Error ($MSE$), Root Mean Squared Error ($RMSE$), and the coefficient of determination ($R^2$).
6. **Inference:** Inferred real estate value for a newly provided median income value.

---

## 📐 Mathematical Formulation

The simple linear regression model is governed by the line equation:

$$\hat{y} = \beta_1 X + \beta_0$$

Where:
* $\hat{y}$ is the predicted median house value (`MedHouseVal`).
* $X$ is the input median income (`MedInc`).
* $\beta_1$ is the slope coefficient (rate of change).
* $\beta_0$ is the $y$-intercept (baseline constant).

### Fitted Model Equation:
$$\text{MedHouseVal} \approx 0.4194 \times \text{MedInc} + 0.4445$$

* **Slope ($\beta_1 \approx 0.4194$):** For every $\$10,000$ increase in district median income, median house prices rise by approximately $\$41,940$.
* **Intercept ($\beta_0 \approx 0.4445$):** The theoretical baseline property value when district income is zero is approximately $\$44,450$.

---

## 📈 Evaluation Results & Analysis

| Metric | Formula | Test Result | Interpretation |
| :--- | :--- | :--- | :--- |
| **MAE** | $\frac{1}{n}\sum \|y_i - \hat{y}_i\|$ | $\approx 0.6277$ | On average, predictions deviate from actual property values by $\approx \$62,770$. |
| **MSE** | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ | $\approx 0.7091$ | Penalizes larger outlier errors more heavily than small errors. |
| **RMSE** | $\sqrt{\text{MSE}}$ | $\approx 0.8421$ | Error measured in the original target units ($\approx \$84,210$). |
| **$R^2$ Score** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | $\approx 0.4589$ | **$45.9\%$** of variance in housing values is explained solely by median income. |

---

## 🧪 Inference on Unseen Sample

* **Input Median Income:** $\$60,000$ (`MedInc = 6.0`)
* **Predicted Target Value:** $2.9609$ ($\times \$100,000$)
* **Estimated Market Value:** **$\$296,090.00$**

---

## ⚠️ Limitations & Future Work

1. **Single-Feature Limitation:** Property values depend on multi-dimensional variables (e.g., geographic coordinates, proximity to ocean, age of housing, rooms per household).
2. **Artificial Target Cap:** The California dataset caps target values at $5.0$ ($\$500,000$), introducing residual horizontal lines for high-value homes.
3. **Future Extension:** Expanding to **Multiple Linear Regression** or non-linear models (e.g., Random Forests, Gradient Boosting) to better capture complex demographic interactions.

---

## 🚀 How to Run the Notebook

### Prerequisites
Make sure Python 3.8+ is installed along with the required libraries:
```bash
pip install numpy pandas matplotlib scikit-learn
```

### Execution
Run the cells sequentially in **Jupyter Notebook** or **Google Colab**:
```bash
jupyter notebook simple_house_price_prediction.ipynb
```