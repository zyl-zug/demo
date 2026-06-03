import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
# Add this import near the top with the others
from xgboost import XGBClassifier

# 1. Load data 
df = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='latin-1')
# print(df.shape)
# print(df.columns.tolist())  # to view all columns in a list . checking the exact column names so you can copy them correctly into your features list.
# select features 
features = [
    'Shipping Mode',                      # dominant predictor — 0.37 to 0.95 range
    'Days for shipment (scheduled)',       # only numeric that matters
    'Type',                               # small signal — PAYMENT vs TRANSFER
    'Category Name',                      # Golf Bags 69% vs Men's Golf Clubs 48%
    'Order Region',
    'Market',
    'Customer Segment',
    'Order Item Quantity',
    'Order Item Product Price',
]

target = 'Late_delivery_risk'

df_model = df[features + [target]].dropna()
print(df_model.shape)
print(df_model[target].value_counts())  #.value_counts() :counts how many rows are 0 and how many are 1
# This tells you your class balance — exactly what your prof covered in Day 3. If it were 1: 99000 and 0: 1000,
# you'd have a badly imbalanced dataset and the model would be useless. Here it's roughly 55/45 so you're fine — no special treatment needed

# ── Split ────────────────────────────────────────────────────────────
X = df_model[features]
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 2. Define numberic vs. categorical
# Replace your features list in train_model.py with this:

numeric_cols = [
    'Days for shipment (scheduled)',
    'Order Item Quantity',
    'Order Item Product Price',
]

categorical_cols = [
    'Shipping Mode',
    'Type',
    'Category Name',
    'Order Region',
    'Market',
    'Customer Segment',
]
# 3. build pipeline 
# ── Shared preprocessor ──────────────────────────────────────────────
# ── Numeric pipeline ─────────────────────────────────────────────────
# Handles the 3 number columns: Quantity, Scheduled Days, Profit Ratio
numeric_pipe = Pipeline([

    # Step 1 — Imputation . `SimpleImputer` — the "fill in the blanks" guy
    # Fills any missing numbers with the MEDIAN of that column
    # Median is preferred over mean because it's robust to outliers
    # e.g. one order with profit ratio = 9999 won't skew the fill value
    ('impute', SimpleImputer(strategy='median')),

    # Step 2 — Scaling
    # Converts numbers to a standard range (mean=0, std=1). StandardScaler converts everything to the **same ruler**:
    # e.g. Quantity (1–65) and Profit Ratio (0.0–1.0) are now comparable
    # Random Forest doesn't strictly need this but it's good practice
    ('scale',  StandardScaler()),
])

# ── Categorical pipeline ─────────────────────────────────────────────
# Handles the 3 text columns: Shipping Mode, Order Region, Category Name
categorical_pipe = Pipeline([

    # Step 1 — Imputation
    # Fills any missing text with the MOST FREQUENT value in that column
    # e.g. if Shipping Mode has a blank, fill with "Standard Class"
    # Can't use median for text — most_frequent is the text equivalent
    ('impute', SimpleImputer(strategy='most_frequent')),

    # Step 2 — One-hot encoding: `OneHotEncoder` — the "translator" guy
    # Converts text categories to 0/1 columns the model can read
    # e.g. Shipping Mode → [is_Standard, is_FirstClass, is_SameDay, is_SecondClass]
    # handle_unknown='ignore' — what if a new customer writes "Drone Delivery" which you've never seen before? Don't crash. Just tick nothing
    # sparse_output=False — give me a normal readable table, not a compressed format.
    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

# ── ColumnTransformer ────────────────────────────────────────────────
#  the **manager** who routes orders to the right station
# Think of it as a traffic controller:
#   numeric_cols  → go to numeric_pipe
#   categorical_cols → go to categorical_pipe
# Then reassembles everything into one wide table for the model
preprocessor = ColumnTransformer([
    ('num', numeric_pipe,     numeric_cols),     # 3 cols → scaled numbers
    ('cat', categorical_pipe, categorical_cols), # 3 cols → 0/1 encoded columns
])

# ── Full model pipeline ──────────────────────────────────────────────
# Chains preprocessor + classifier into ONE object
# This is the key design decision:
#   - joblib.dump() saves THE WHOLE THING as model.pkl
#   - When Streamlit calls model.predict_proba(new_order),
#     it automatically runs impute → scale → encode → predict
#   - You never have to manually preprocess in app.py
# ── Two models in a dictionary — train all, compare AUCs ─────────────
models = {
    'RandomForest': Pipeline([
        ('prep', preprocessor),
        ('clf',  RandomForestClassifier(
            n_estimators = 200,
            random_state = 42,
            n_jobs       = -1,
        )),
    ]),
    # Random Forest classifier
    # n_estimators=200 → builds 200 decision trees, averages their votes
    # More trees = more stable predictions, slower training
    # random_state=42  → reproducible results (same output every run)
    # n_jobs=-1        → use all CPU cores in parallel (faster training)

    'XGBoost': Pipeline([
        ('prep', preprocessor),
        ('clf',  XGBClassifier(
            n_estimators = 200,
            random_state = 42,
            eval_metric  = 'logloss',
            verbosity    = 0,
            n_jobs       = -1,
        )),
    ]),
}

# ── Train + score each ───────────────────────────────────────────────
results = {}
trained = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    results[name] = auc
    trained[name] = pipe
    print(f"{name:15s}  AUC-ROC = {auc:.4f}")

# ── Pick winner — highest AUC ────────────────────────────────────────
winner_name = max(results, key=results.get)
winner_pipe = trained[winner_name]
print(f"\nWinner: {winner_name}  (AUC = {results[winner_name]:.4f})")

# ── Save winning model + schema ──────────────────────────────────────
joblib.dump(winner_pipe, 'model.pkl')

# Save a small pre-aggregated overview file for the deployed app
overview_summary = df_model.copy()
overview_summary['order_value'] = (
    overview_summary['Order Item Product Price'] *
    overview_summary['Order Item Quantity']
)
overview_summary.to_parquet('overview_data.parquet', index=False)
print("Saved: overview_data.parquet")

schema = {
    'features'         : features,
    'numeric_cols'     : numeric_cols,
    'categorical_cols' : categorical_cols,
    'model_name'       : winner_name,
    'leaderboard'      : {k: round(v, 4) for k, v in results.items()},
    'categories': {
        col: sorted(df_model[col].dropna().unique().tolist())
        for col in categorical_cols
    },
    'numeric_ranges': {
        col: {
            'min'  : float(df_model[col].min()),
            'max'  : float(df_model[col].max()),
            'mean' : float(df_model[col].mean()),
        }
        for col in numeric_cols
    }
}

with open('schema.json', 'w') as f:
    json.dump(schema, f, indent=2)

print(f"\nSaved: model.pkl ({winner_name}) + schema.json")
# model.pkl — trained Random Forest pipeline, ready to load
# schema — the JSON widget instruction manual for Streamlit
# train_model — your Python script (keep this, it's your reproducibility proof

# notes for debugging 
# a. found "days for shipping (schduled)" is a leakage column 
# Leakage = using information that wouldn't exist yet in real life when the prediction needs to be made.
# The model trained on leaked columns learns a shortcut — "oh, Days for shipping (real) = 7 and scheduled = 4? That's late." It gets a great training score but completely fails in production because that column doesn't exist for a brand new order. 
# It's like letting a student see the answer key during the exam — great score, zero learning.