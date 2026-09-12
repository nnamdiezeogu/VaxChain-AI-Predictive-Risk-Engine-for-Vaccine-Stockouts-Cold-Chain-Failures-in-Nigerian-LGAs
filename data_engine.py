import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. Load the real GRID3 dataset downloaded from GRID3 Data Hub
try:
    # Adjust filename if your downloaded version is different
    df_real_grid3 = pd.read_csv("GRID3_Nigeria_Health_Facilities.csv")
except FileNotFoundError:
    print("Error: Please download the real GRID3 CSV and save it as 'GRID3_Nigeria_Health_Facilities.csv'")
    exit()

# 2. Filter strictly for the KEDCO footprint PHCs
# Standard GRID3 coloumns typically use 'state', 'lga', 'facility_level', 'latitude', 'longitude'
target_states = ["Kano", "Katsina", "Jigawa"]

# Lowercase strings or strip spaces to prevent mismatch issues during matching
df_real_grid3['state']=df_real_grid3["state"].str.strip()
df_real_grid3['facility_level']=df_real_grid3["facility_level"].str.strip()

# Apply the strict filters you highlighted
df_filtered_phc = df_real_grid3[
    (df_real_grid3["state"].isin(target_states)) &
    (df_real_grid3["facility_level"].str.contains('primary', case=False, na=False))
].copy()

print(f" Extracted {len(df_filtered_phc)} actual GRID3 PHCs across Kano, Katsina, and Jigawa.")

# Apply Sentinel facility sampling to keep dashboard smooth and responsive
# Group by state and LGA and grab 2 facilities per LGA
df_sampled_phc = df_filtered_phc.groupby(['state', 'lga']).sample(n=2, random_state=42, replace=True).reset_index(drop=True)
df_sampled_phc = df_sampled_phc.drop_duplicates(subset=['latitude', 'longitude']) # Safeguard coordinates

print(f" Randomly selected {len(df_sampled_phc)} sentinel GRID3 PHCs (max 2 per LGA) for demonstration optimization.")

# 3. Map real facilities into KEDCO environmental risk variables
kedco_meta = {
    "Kano": {"kedco_grid_availability": 0.35, "transit_days": 4, "base_demand": 45, "cce_type": "Ice_Lined"},
    "Katsina": {"kedco_grid_availability": 0.20, "transit_days": 8, "base_demand": 25, "cce_type": "SSD_Solar"},
    "Jigawa": {"kedco_grid_availability": 0.30, "transit_days": 6, "base_demand": 20, "cce_type": "SSD_Solar"}
}

# Add KEDCO specific logistics indicators to the real cordinates matrix
phc_registry = []
for idx, row in df_sampled_phc.iterrows():
    state = row['state']
    meta = kedco_meta[state]

    phc_registry.append({
        "facility_id": row.get('nhfr_uid', f"PHC_{idx:04d}"),
        "facility_name": row.get('facility_name', f"{row['lga']} PHC Node"),
        "lga": row['lga'],
        "state": state,
        "latitude": row['latitude'],
        "longitude": row['longitude'],
        "cce_type": meta["cce_type"],
        "grid_reliability": meta["kedco_grid_availability"],
        "transit_days": meta["transit_days"],
        "base_demand": meta["base_demand"]
    })

df_phc = pd.DataFrame(phc_registry)

# 4. 30-Day Ledge Simulation Loop
start_date = datetime(2026, 8, 1)
days_to_simulate = 30
ledger_records = []

for idx, phc in df_phc.iterrows():
    stock = {"BCG": phc["base_demand"] * 6, "Penta": phc["base_demand"] * 6, "Measles": phc["base_demand"] * 6}
    reorder_point = phc["base_demand"] * 2
    fridge_temp = 4.0

    for day in range(days_to_simulate):
        current_date = start_date + timedelta(days=day)

        # A. Cold Chain Module (Anchored on KEDCO grid reliability)
        grid_fail = np.random.random() > phc["grid_reliability"]
        if grid_fail:
            temp_rise = np.random.normal(loc=0.9, scale=0.3) if phc["cce_type"] == "SDD_Solar" else np.random.normal(loc=3.6, scale=1.3)
            fridge_temp = min(24.0, fridge_temp + max(0.1, temp_rise))
        else:
            fridge_temp = max(2.0, fridge_temp - np.random.normal(loc=2.8, scale=0.6))

            # B. Vaccine Inventory Module
            for antigen in stock.keys():
                daily_administered = np.random.poisson(lam=phc["base_demand"])
                stock[antigen] = max(0, stock[antigen] - daily_administered)
                is_stockout = 1 if stock[antigen] == 0 else 0

                # Heat spoilage rules based on sustained temperatue breaks
                if fridge_temp > 12.0 and np.random.random() < 0.20:
                    stock[antigen] = 0
                    is_stockout = 1

                # Last mile logistics fulfilment
                if stock[antigen] <= reorder_point:
                    if np.random.random() < (1.0 / phc["transit_days"]):
                        stock[antigen] += int(phc["base_demand"] * 8)

                ledger_records.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "facility_id": phc["facility_id"],
                    "facility_name": phc["facility_name"],
                    "lga": phc["lga"],
                    "state": phc["state"],
                    "latitude": phc["latitude"],
                    "longitude": phc["longitude"],
                    "antigen_type": antigen,
                    "current_stock_level": stock[antigen],
                    "fridge_temperature_celsius": round(fridge_temp, 2),
                    "grid_status_operational": 0 if grid_fail else 1,
                    "is_stockout": is_stockout
                    })

df_vax_ledger = pd.DataFrame(ledger_records)
df_vax_ledger.to_csv("filtered_phc_vax_ledger.csv", index=False)
print(f"Generated {len(df_vax_ledger)} data rows focusing strictly on KEDO PHCs (Kano, Katsina, Jigawa).")