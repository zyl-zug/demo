# 🚚 Late Delivery Risk Predictor

End-to-end ML pipeline predicting whether an order will arrive late, 
deployed as a Streamlit app.

**Live demo:** [https://zyl-supply-chain-risk.streamlit.app/]

## Business problem
Predict probability of late delivery at order time, so logistics managers 
can prioritise interventions before shipments fail.

## Dataset
DataCo Global Supply Chain — 180K orders, 53 columns.

## Key findings
- **First Class shipping is late ~95% of the time** — not because it's slow, 
  but because the 1-day promise exceeds operational capacity
- **Shipping Mode is the dominant predictor** — geography and customer type 
  barely matter
- **Data leakage trap**: many public solutions report 97% accuracy by using 
  post-delivery columns. Excluding them gives an honest 0.74 AUC ceiling.

## Tech stack
- scikit-learn (preprocessing pipelines)
- XGBoost vs Random Forest comparison
- SHAP for per-order explanations
- Streamlit for the UI

## Run locally
```bash
pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```
