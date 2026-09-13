import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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
model = RandomForestClassifier(n_estimators=150, class_weight='balanced', max_depth=10, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "vaxchain_rf_model.pkl")

# 3. Model Evaluation
# Make predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Classification Report
print("\nClassification Report:")
report = classification_report(y_test, y_pred)
print(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - VaxChain-AI')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save Feature Importance Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance - VaxChain-AI')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# Save Results to File
with open("model_performance.txt", "w", encoding="utf-8") as f:
    f.write("==== VaxChain-AI Model Performance ====\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nFeature Importance:\n")
    f.write(feature_importance.to_string())

print("\nPerformance results saved to 'model_performace.txt'")
print("Confusion matrix saved as 'confusion_matrix.png'")
print("Feature importance saved as 'feature_importance.png'")

# 4. Compile Dashboard State Layers
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