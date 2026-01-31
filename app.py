import streamlit as st
import pandas as pd
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="AI Traffic Intelligence Dashboard",
    page_icon="🚦",
    layout="wide"
)

# -------------------------------
# Title
# -------------------------------
st.title("🚦 AI‑Powered Traffic Queue Analysis & Rule Violation Detection")
st.caption("Video Analytics Dashboard using YOLO + AI Models")

# -------------------------------
# File Check
# -------------------------------
required_files = ["traffic.mp4", "yolo_output.mp4", "traffic_data.csv"]
for f in required_files:
    if not os.path.exists(f):
        st.error(f"Missing file: {f}")
        st.stop()

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("traffic_data.csv")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("⚙️ Dashboard Controls")

confidence_filter = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

if "confidence" in df.columns:
    df = df[df["confidence"] >= confidence_filter]

st.sidebar.markdown("---")
junction = st.sidebar.text_input("Junction Name", "Main City Signal")
lanes = st.sidebar.number_input("Number of Lanes", 1, 8, 4)

# -------------------------------
# KPI Section
# -------------------------------
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("🚗 Vehicles", df["vehicle_id"].nunique())
k2.metric("📦 Queue Length", df[df["in_queue"]].vehicle_id.nunique())
k3.metric("🚨 Violations", df[df["red_light_violation"]].vehicle_id.nunique())
k4.metric("⚠ Rash Driving", df[df["rash_driving"]].vehicle_id.nunique())
k5.metric("📊 Frames", df["frame"].max())

st.markdown("---")

# -------------------------------
# Video Section
# -------------------------------
st.subheader("🎥 Video Analytics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📥 Raw Traffic Input Video")
    st.video("traffic.mp4")

with col2:
    st.markdown("### 🤖 YOLO Output Video")
    st.video("yolo_output.mp4")

st.markdown("---")

# -------------------------------
# Analytics Section
# -------------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("📊 Tracked Vehicle Data")
    st.dataframe(df.head(300), use_container_width=True)

with right:
    st.subheader("📈 Vehicle Types")
    st.bar_chart(df["vehicle_type"].value_counts())

# -------------------------------
# Tabs Section
# -------------------------------
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📦 Queue Analysis",
    "🚨 Violations",
    "📊 Traffic Flow",
    "🚗 Vehicle Types",
    "📁 Reports"
])

# ---- Queue Tab ----
with tab1:
    st.subheader("Queue Per Lane (Mock Logic)")
    lane_data = pd.DataFrame({
        "Lane": [f"Lane {i+1}" for i in range(lanes)],
        "Queue Length": [df[df["in_queue"]].vehicle_id.nunique()//lanes for _ in range(lanes)]
    })
    st.table(lane_data)

    st.metric("Total Queue Length", df[df["in_queue"]].vehicle_id.nunique())

# ---- Violations Tab ----
with tab2:
    st.subheader("Violation Records")
    violation_df = df[(df["red_light_violation"]) | (df["rash_driving"])]
    st.dataframe(
        violation_df[["frame","vehicle_id","vehicle_type","red_light_violation","rash_driving"]],
        use_container_width=True
    )

# ---- Traffic Flow Tab ----
with tab3:
    st.subheader("Traffic Flow Over Time")
    flow = df.groupby("frame")["vehicle_id"].nunique().reset_index()
    flow.columns = ["Frame", "Vehicle Count"]
    st.line_chart(flow.set_index("Frame"))

# ---- Vehicle Types Tab ----
with tab4:
    st.subheader("Vehicle Distribution")
    st.bar_chart(df["vehicle_type"].value_counts())

# ---- Reports Tab ----
with tab5:
    st.subheader("Reports & Downloads")
    st.download_button(
        "📄 Download Full CSV Report",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="traffic_report.csv"
    )

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("AI‑Powered Traffic System | YOLO + Video Analytics + Streamlit Dashboard")
