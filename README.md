# Machine-Learning-Based-Price-Fairness-Analysis-for-Laptop-E-Commerce

When shopping for a laptop online, it is hard to know if the listed price is actually fair. Prices vary widely across brands and retailers, and without deep technical knowledge, most consumers cannot judge whether a laptop is overpriced relative to its specs. This project tackles that problem directly.

It builds a machine learning pipeline that learns the relationship between hardware specifications and market price, then uses that to compute a Price Fairness Score for any laptop listing. The score measures the gap between what a laptop actually costs and what the model predicts it should cost based on its specs -- flagging listings as Overpriced, Fairly Priced, or Underpriced. It also includes a Streamlit demo app where users can paste any product description and get an instant fairness verdict.

---

## Project Structure

```
Machine-Learning-Based-Price-Fairness-Analysis-for-Laptop-E-Commerce/
|-- notebooks/
|   |-- EDA.ipynb
|   |-- feature_engineering.ipynb
|   |-- modeling.ipynb
|-- streamlit_app/
|   |-- app.py
|   |-- requirements.txt
|   |-- .streamlit/
|       |-- secrets.toml
|-- data/
|   |-- raw/
|       |-- laptop_cleaned2.csv
|-- requirements.txt
|-- README.md
|-- FinalProject_doc.pdf
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ManjushaMotamarry/Machine-Learning-Based-Price-Fairness-Analysis-for-Laptop-E-Commerce.git
cd Machine-Learning-Based-Price-Fairness-Analysis-for-Laptop-E-Commerce
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Notebooks

Run the notebooks in this order:

```
1. notebooks/EDA.ipynb
2. notebooks/feature_engineering.ipynb
3. notebooks/modeling.ipynb
```

Each notebook is self-contained with markdown cells explaining each step. The modeling notebook trains all three models, runs SHAP analysis, computes the Price Fairness Score, and saves the tuned XGBoost model to `xgb_tuned_model.pkl`.

---

## Running the Streamlit App

# 1. From project root, activate venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 2. Navigate to app folder and install dependencies
cd streamlit_app
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py


The app will open at `http://localhost:8501` in your browser.

### How to use it

1. Paste any laptop product description (e.g. from Amazon or a retailer's website) into the text box
2. Enter the listed price in INR
3. Click **Check Fairness**
4. The app extracts specs using regex-based parsing, runs inference with the trained XGBoost model, and returns a predicted price and fairness verdict (Overpriced / Fairly Priced / Underpriced)

---

## Dataset

Siddiqui, F. (2024). Laptop Sales Price Prediction Dataset. Kaggle.
https://www.kaggle.com/datasets/siddiquifaiznaeem/laptop-sales-price-prediction-dataset-2024/data

---

## Dependencies

- Python 3.11
- scikit-learn
- xgboost
- shap
- pandas
- numpy
- matplotlib
- seaborn
- streamlit
- joblib
