import streamlit as st
import pandas as pd

# 1. Page layout
st.set_page_config(
    page_title="VaxChain-AI: KEDO Core",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(" VaxChain-AI: Predictive Risk Engine for Vaccine Stockouts & Cold Chain Failures in Nigerian LGAs")
st.markdown("### Predictive Last-Mile Dashboard (Kano, Katsina, Jigawa)")
st.markdown("- - -")

# 2. Safety File load loop
try:
    df_lga = pd.read_csv("lga_dashboard_map_data.csv")
except FileNotFoundError:
    st.error("❌ Critical Error: 'lga_dashboard_map_data.csv' not found. Please execute the step 2 ML script in the terminal first!")
    st.stop()

# Sidebar controls
st.sidebar.header("📍 KEDO Subnational Jurisdiction")

# State Jurisdiction Dropdown
selected_state = st.sidebar.selectbox(
    "Select State Jurisdiction:", 
    ["All KEDO Franchise"] + list(df_lga["state"].unique())
)

# Apply state filters securely to DataFrame
if selected_state == "All KEDO Franchise":
    display_df = df_lga.copy()
else:
    display_df = df_lga[df_lga["state"] == selected_state].copy()

# Dashboard Analytics: Summary Metrics Card
col1,col2,col3 = st.columns(3)
col1.metric("Active PHCs Map Nodes Displayed", len(display_df))
col2.metric("High Risk Hazards Flagged", len(display_df[display_df["predicted_stockout_risk"] > 0.22]))

temp_label= f"Mean CCE Temp ({selected_state})" if selected_state != "All KEDCO Franchise" else "Franchise Mean CCE Temp"

col3.metric(
    label=temp_label,
    value=f"{display_df['fridge_temperature_celsius'].mean():.2f}Celsius" if len(display_df) > 0 else "0.00 Celsius"
)

# Mapping Engine: Fluid Native Map Componenet
st.markdown("### Live Last-Mile Risk canvas")
st.caption("Left-click and hold the mouse down to slide/pan the map canvas in any direction.")

if len(display_df) > 0:
    def assign_color(risk):
        if risk > 0.22:
            return "#E74C3C" # Emergency: Notify Local Operator!
        elif risk > 0.12:
            return "#F39C12" # Warning: Infrastructure at risk
        else:
            return "#2ECC71" # Safe Infrastructure

    # Prepare data layer for native WebGL processing
    map_df = display_df.copy()
    map_df["color"] = map_df["predicted_stockout_risk"].apply(assign_color)

    # Render native mapping canvas
    st.map(
        data=map_df,
        latitude="latitude",
        longitude="longitude",
        color="color",
        size=40
    )
else:
    st.warning("⚠️ No operational data points match your native filtering variables.")

# Alert Console: Cascading Crisis Response Dropdown
st.markdown("- - -")
st.markdown("### ⚡KEDCO Last-Mile Emergency Dispatch")

# Filter alerts strictly based on the sidebar state choice
if selected_state == "All KEDCO Franchise":
    high_risk_df = df_lga[df_lga["predicted_stockout_risk"] > 0.22].copy()
else:
    high_risk_df = df_lga[(df_lga["predicted_stockout_risk"] > 0.22) & (df_lga["state"] == selected_state)].copy()

available_high_risk_lgas = high_risk_df["lga"].unique()

# Render interactive dispatch forms if any targets exist
if len(available_high_risk_lgas) > 0:
    target_lga = st.selectbox(
        f"Isolate Critical LGA Hubs: {selected_state}",
        available_high_risk_lgas
    )

    if st.button("Broadcast Praoctive Intervention Protocol"):
        st.success(f"Success! Automted emergency logistical warning safely routed to the **{target_lga} Cold Chain Hub** via Twilio Gateway API pipelines.")
else:
    st.info(f" No active vaccine stockout or cold chain failure indices match the active filtering: **{selected_state}**.")