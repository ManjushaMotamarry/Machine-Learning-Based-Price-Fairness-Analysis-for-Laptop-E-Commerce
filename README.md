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
|-- run.sh
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

Make sure to select the kernel as the previosuly created venv

Run the notebooks in this order:

```
1. notebooks/EDA.ipynb
2. notebooks/feature_engineering.ipynb
3. notebooks/modeling.ipynb
```

Each notebook is self-contained with markdown cells explaining each step. The modeling notebook trains all three models, runs SHAP analysis, computes the Price Fairness Score, and saves the tuned XGBoost model to `xgb_tuned_model.pkl`.

---

## Expected structure after running the notebooks

```
Machine-Learning-Based-Price-Fairness-Analysis-for-Laptop-E-Commerce/
|-- notebooks/
|   |-- EDA.ipynb
|   |-- feature_engineering.ipynb
|   |-- modeling.ipynb
|   |-- xgb_tuned_model.pkl
|-- streamlit_app/
|   |-- app.py
|   |-- requirements.txt
|   |-- xgb_tuned_model.pkl
|   |-- .streamlit/
|       |-- secrets.toml
|-- data/
|   |-- raw/
|       |-- laptop_cleaned2.csv
|   |-- processed/
|       |-- laptop_eda.csv
|       |-- laptop_label_encoded.csv
|       |-- laptop_onehot_encoded.csv
|       |-- price_fairness_scores.csv
|-- requirements.txt
|-- README.md
|-- FinalProject_doc.pdf
```

## Running the Streamlit App

Use the provided script to set up and launch the app in one step:

```bash
bash run.sh
```

This will create a virtual environment, install all dependencies, and launch the app at `http://localhost:8501`.

If you prefer to run manually:

```bash
# 1. Activate venv (from project root)
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies and run
pip install -r streamlit_app/requirements.txt
cd streamlit_app
python -m streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.


### How to use it

1. Paste any laptop product description into the text box — use listings from **Amazon.in** (not Amazon.com). The model is trained on INR prices, so USD listings will give incorrect results.
2. Enter the listed price in **INR** (e.g. 90000)
3. Click **Check Fairness**
4. The app extracts specs using regex-based parsing, runs inference with the trained XGBoost model, and returns a predicted price and fairness verdict (Overpriced / Fairly Priced / Underpriced)

**Example input:**

```
Brand: ASUS
Model Name: ASUS Vivobook 14
Screen Size: 14 Inches
Hard Disk Size: 512 GB
CPU Model: Ryzen AI 5
RAM Memory Installed Size: 16 GB
Operating System: Windows 11 Home
Graphics Card Description: Integrated
```

**Listed Price:** 90000
```
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
