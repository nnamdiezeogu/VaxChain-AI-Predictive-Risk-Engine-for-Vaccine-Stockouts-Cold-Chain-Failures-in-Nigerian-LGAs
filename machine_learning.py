import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Ingest Generated Filtered PHC Ledger
df = pd.read_csv("filtered_phc_vax_ledger.csv")
df = df.sort_values(by=["facility_id", "antigen_type", "date"])

# 2. Feature Engineering (per Facility Granularity)
df["rolling_avg_consumption"] = df.groupby(["facility_id", "antigen_type"])["current_stock_level"].diff().fillna(0).abs()
df["rolling_avg_consumption"] = df.groupby(["facility_id", "antigen_type"])["rolling_avg_consumption"].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["rolling_max_temp"] = df.groupby(["facility_id", "antigen_type"])["fridge_temperature_celsius"].transform(lambda x: x.rolling(3, min_periods=1).max())
df["target_stockout_3days"] = df.groupby(["facility_id", "antigen_type"])["is_stockout"].shift(-3).fillna(0).astype(int)

features = ["current_stock_level", "fridge_temperature_celsius", "grid_status_operational", "rolling_avg_consumption", "rolling_max_temp"]
X = df[features]
y = df["target_stockout_3days"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "vaxchain_rf_model.pkl")

# 3. Compile Dashboard State Layers
df["predicted_stockout_risk"] = model.predict_proba(df[features])[:, 1]
latest_date = df["date"].max()
df_latest = df[df["date"] == df["date"]]

lga_dashboard_data = df_latest.groupby(["lga", "state", "latitude", "longitude"]).agg({
    "current_stock_level": "sum",
    "fridge_temperature_celsius": "mean",
    "predicted_stockout_risk": "mean"
}).reset_index()

lga_dashboard_data.to_csv("lga_dashboard_map_data.csv", index=False)
print("KEDCO franchsie aggregated map dataset saved to 'lga_dashboard_map_data.csv'")