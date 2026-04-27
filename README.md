<div align="center">

<img src="https://capsule-render.vercel.app/api?type=cylinder&color=gradient&customColorList=2,3,12&height=180&section=header&text=📐%20Regression%20Algorithms&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=45&desc=Machine%20Learning%20Assignment%20•%20Python%20•%20Scikit-Learn&descAlignY=68&descAlign=50" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Regression-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-4%20Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Pandas](https://img.shields.io/badge/Pandas-DataFrames-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge)](https://matplotlib.org)

<br/>

> **From salary prediction to housing prices — a hands-on exploration of 5 regression algorithms, regularization techniques, and model evaluation.**

<br/>

[📖 Overview](#-overview) • [🧪 Algorithms](#-algorithms-covered) • [📊 Results](#-model-performance) • [📁 Structure](#-project-structure) • [🚀 Quick Start](#-quick-start) • [📚 Concepts](#-key-concepts)

</div>

---

## 📖 Overview

This assignment is a structured, practical exploration of **Regression in Machine Learning** — going from the simplest straight-line fit to regularized models that handle high-dimensional, noisy data.

Each algorithm is implemented in its own dedicated Jupyter notebook with:
- ✅ Clean, well-commented code
- ✅ Real-world datasets
- ✅ Evaluation metrics (R² Score & MSE)
- ✅ Visualizations and plots
- ✅ Final cross-model comparison

---

## 🧪 Algorithms Covered

### 1️⃣ Simple Linear Regression
> `notebooks/1_simple_linear_regression.ipynb`

**Dataset:** `salary_dataset.csv` — Years of Experience → Salary  
**Concept:** Fits a single straight line `y = mx + b` to model the relationship between one input feature and a continuous target.

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

- **Feature:** YearsExperience
- **Target:** Salary
- **Output:** Regression line plot + R² / MSE

---

### 2️⃣ Multiple Linear Regression
> `notebooks/2_multiple_linear_regression.ipynb`

**Dataset:** `housing_dataset.csv` — Area, Bedrooms, Bathrooms → Price  
**Concept:** Extends linear regression to multiple input features: `y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ`

```python
# Features: area, bedrooms, bathrooms → Target: price
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']
```

- **Features:** Area (sq ft), Bedrooms, Bathrooms
- **Target:** House Price
- **Output:** Coefficient analysis + predictions

---

### 3️⃣ Polynomial Regression
> `notebooks/3_polynomial_regression.ipynb`

**Dataset:** `polynomial_dataset.csv`  
**Concept:** Captures non-linear relationships by transforming features into polynomial terms before fitting a linear model.

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

- **Use case:** Curved, non-linear data patterns
- **Output:** Polynomial curve fit visualization

---

### 4️⃣ Ridge & Lasso Regression (Regularization)
> `notebooks/4_ridge_lasso_regression.ipynb`

**Concept:** Regularized regression models that penalize large coefficients to prevent overfitting.

| | Ridge (L2) | Lasso (L1) |
|---|---|---|
| **Penalty** | Sum of squared coefficients | Sum of absolute coefficients |
| **Effect** | Shrinks all coefficients | Can zero out coefficients (feature selection) |
| **Best for** | Multicollinearity | Sparse feature sets |
| **Alpha used** | `1.0` | `0.1` |

```python
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
```

---

## 📊 Model Performance

Results from `results/model_scores/model_comparison.csv`:

| Rank | Model | R² Score | MSE |
|---|---|---|---|
| 🥇 | **Ridge Regression** | **0.9952** | 15,112,917,478 |
| 🥈 | **Lasso Regression** | **0.9944** | 17,605,188,954 |
| 🥉 | Simple Linear | 0.9289 | 57,069,614 |
| 4️⃣ | Polynomial | 0.7666 | 11,813,379,747 |

> 💡 **Key Insight:** Ridge and Lasso significantly outperform baseline models on the housing dataset, demonstrating the power of regularization for controlling model complexity.

---

## 🧠 Key Concepts

<details>
<summary><b>📐 What is R² Score?</b></summary>

R² (coefficient of determination) measures how well the model explains variance in the target variable.
- **R² = 1.0** → Perfect fit
- **R² = 0.0** → Model is no better than the mean
- **R² < 0** → Model is worse than the mean

</details>

<details>
<summary><b>📉 What is MSE?</b></summary>

Mean Squared Error (MSE) measures the average squared difference between predicted and actual values.
- **Lower MSE = Better model**
- Sensitive to outliers (squares errors)
- Useful for comparing models on the same dataset

</details>

<details>
<summary><b>⚖️ Bias-Variance Tradeoff</b></summary>

- **High Bias (Underfitting):** Model is too simple, misses patterns — e.g., fitting a line to curved data
- **High Variance (Overfitting):** Model memorizes training data, fails on new data
- **Regularization (Ridge/Lasso):** Reduces variance by penalizing complexity — improving generalization

</details>

<details>
<summary><b>🔢 When to use which regression?</b></summary>

| Scenario | Use |
|---|---|
| One feature, linear relationship | Simple Linear |
| Multiple features, linear | Multiple Linear |
| Curved/non-linear pattern | Polynomial |
| Many features, prevent overfitting | Ridge |
| Many features, want feature selection | Lasso |

</details>

---

## 📁 Project Structure

```bash
Regression_Assignment/
│
├── 📂 data/
│   ├── salary_dataset.csv          # YearsExperience → Salary (25 records)
│   ├── housing_dataset.csv         # Area/Bedrooms/Bathrooms → Price (20 records)
│   └── polynomial_dataset.csv      # Non-linear synthetic data
│
├── 📂 notebooks/
│   ├── 1_simple_linear_regression.ipynb     # Salary prediction
│   ├── 2_multiple_linear_regression.ipynb   # Housing price prediction
│   ├── 3_polynomial_regression.ipynb        # Curve fitting
│   └── 4_ridge_lasso_regression.ipynb       # Regularization comparison
│
├── 📂 results/
│   ├── graphs/
│   │   └── model_comparison.png             # Visual performance comparison
│   └── model_scores/
│       └── model_comparison.csv             # R² and MSE for all models
│
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/regression-assignment.git
cd regression-assignment
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Launch Jupyter Lab

```bash
jupyter lab
```

### 4. Run Notebooks in Order

```
notebooks/1_simple_linear_regression.ipynb     ← Start here
notebooks/2_multiple_linear_regression.ipynb
notebooks/3_polynomial_regression.ipynb
notebooks/4_ridge_lasso_regression.ipynb       ← Final comparison
```

> 📌 Each notebook is **self-contained** and can be run independently.

---

## 📦 Dependencies

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![JupyterLab](https://img.shields.io/badge/JupyterLab-F37626?style=flat-square&logo=jupyter&logoColor=white)

</div>

---

## 📚 Learning Outcomes

By working through this assignment, the following skills are demonstrated:

- [x] Implementing regression algorithms from scratch using Scikit-Learn
- [x] Preprocessing data and splitting train/test sets
- [x] Understanding the mathematics behind each algorithm
- [x] Evaluating models using R² Score and MSE
- [x] Visualizing regression lines, curves, and predictions
- [x] Understanding regularization (Ridge vs Lasso) and when to use each
- [x] Comparing multiple models systematically

---

## 👩‍💻 Author

<div align="center">

**Ananya Mangaj**  
*Machine Learning Assignment — Regression Algorithms*

[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/your-username)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/your-profile)

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=cylinder&color=gradient&customColorList=2,3,12&height=100&section=footer" width="100%"/>

*⭐ Star this repo if it helped you understand regression!*

</div>
