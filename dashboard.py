import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Green Cloud Scheduling Dashboard",
    layout="wide"
)

# ======================
# LOAD DATA
# ======================

final_df = pd.read_csv("final_output.csv")
ci_df = pd.read_csv(r"E:\ML Project\Carbon_Intensity_Data.csv")

# ======================
# TITLE
# ======================

st.title("Carbon-Aware Machine Learning–Based Workload Scheduling for Energy-Efficient Cloud Data Centers")

st.markdown("""
This interactive dashboard explains an **end-to-end Machine Learning–based framework**
for **energy-efficient and carbon-aware cloud workload scheduling**.
""")

# ======================
# SIDEBAR
# ======================

st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Select Section",
    [
        "Introduction & Motivation",
        "Problem Statement",
        "System Architecture",
        "Energy Prediction (ML Models)",
        "Carbon Intensity & Workload Analysis",
        "Green Scheduling Decisions"
    ]
)

# ======================
# SECTION 1: INTRODUCTION
# ======================

if section == "Introduction & Motivation":
    st.header("☁️ Introduction & Motivation")

    st.subheader("What is Cloud Computing?")
    st.markdown("""
    Cloud computing refers to the delivery of computing resources such as servers,
    storage, databases, and applications over the internet.
    Large cloud providers operate massive data centers that continuously execute
    millions of workloads.
    """)

    st.subheader("Why Do Data Centers Consume So Much Energy?")
    st.markdown("""
    - Servers run 24×7 to handle unpredictable workloads  
    - CPUs and memory consume power even when idle  
    - Cooling systems are required to prevent overheating  
    - Networking and storage infrastructure also contribute to power consumption
    """)

    st.subheader("What is Carbon Intensity?")
    st.markdown("""
    Carbon intensity measures how much carbon dioxide (CO₂) is emitted per unit
    of electricity generated (gCO₂/kWh).
    Electricity generated from renewable sources has lower carbon intensity than
    fossil-fuel-based electricity.
    """)

    st.subheader("Why Does Scheduling Affect Carbon Emissions?")
    st.markdown("""
    Carbon emissions depend on *when* a job is executed, not just *how much* energy it uses.

    **Carbon Emission = Energy Consumption × Carbon Intensity**

    Scheduling workloads during low-carbon periods can significantly reduce emissions.
    """)

# ======================
# SECTION 2: PROBLEM STATEMENT
# ======================

if section == "Problem Statement":
    st.header("❗ Problem Statement")

    st.markdown("""
    Modern cloud workload schedulers focus mainly on performance metrics such as
    CPU utilization and response time. They ignore the environmental impact of
    electricity generation and do not consider workload flexibility.
    """)

    st.subheader("Key Challenges")
    st.markdown("""
    - High energy consumption in cloud data centers  
    - Time-varying carbon intensity of electricity grids  
    - No distinction between flexible and non-flexible workloads  
    """)

    st.subheader("Project Objectives")
    st.markdown("""
    This project aims to:
    - Predict energy consumption using classical Machine Learning  
    - Estimate carbon emissions using real grid carbon intensity data  
    - Classify workload flexibility using ML  
    - Make carbon-aware scheduling decisions without using Deep Learning
    """)

# ======================
# SECTION 3: ARCHITECTURE
# ======================

if section == "System Architecture":
    st.header("🏗️ System Architecture")

    st.subheader("Textual Architecture Flow")
    st.code("""
    Input Datasets
        ↓
    Data Preprocessing & Feature Engineering
        ↓
    ML Models (Energy Prediction & Flexibility Classification)
        ↓
    Green Scheduling Score Computation
        ↓
    Scheduling Decision (Run / Delay)
        ↓
    Performance Evaluation
    """)

    st.subheader("Block-Level Explanation")
    st.markdown("""
    **Input Layer:**  
    - UCI Energy Efficiency Dataset  
    - Google Cluster Workload Traces  
    - UK National Grid Carbon Intensity Dataset  

    **ML Layer:**  
    - Random Forest for energy prediction  
    - Decision Tree Classifier for workload flexibility  

    **Decision Layer:**  
    - Green Scheduling Score integrates energy, carbon, and SLA awareness  

    **Evaluation Layer:**  
    - Energy and carbon reduction analysis  
    - Scheduling decision outcomes
    """)

# ======================
# SECTION 4: ENERGY PREDICTION
# ======================

if section == "Energy Prediction (ML Models)":
    st.header("⚡ Energy Consumption Prediction Using ML")

    metrics_df = pd.DataFrame({
        "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
        "MAE": [2.15, 0.40, 0.33],
        "RMSE": [2.97, 0.59, 0.47],
        "R² Score": [0.91, 0.99, 0.99]
    })

    st.subheader("Model Performance Metrics")
    st.dataframe(metrics_df)

    st.markdown("""
    - **MAE:** Average prediction error  
    - **RMSE:** Penalizes large errors  
    - **R²:** Percentage of variance explained
    """)

    st.subheader("📊 Model Comparison (RMSE)")
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    ax1.bar(metrics_df["Model"], metrics_df["RMSE"])
    ax1.set_ylabel("RMSE")
    ax1.set_title("Energy Prediction Accuracy Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(fig1)

    st.subheader("📈 Actual vs Predicted Energy (Random Forest)")
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.scatter(final_df["Predicted_Energy"], final_df["Predicted_Energy"])
    ax2.set_xlabel("Actual Energy")
    ax2.set_ylabel("Predicted Energy")
    ax2.set_title("Actual vs Predicted Energy")
    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(fig2)

# ======================
# SECTION 5: CARBON & WORKLOAD
# ======================

if section == "Carbon Intensity & Workload Analysis":
    st.header("🌱 Carbon Intensity & Workload Analysis")

    st.subheader("Hourly Carbon Intensity")
    carbon_values = ci_df.select_dtypes(include=[np.number]).iloc[:, -1].dropna().values

    fig3, ax3 = plt.subplots(figsize=(5, 3))
    ax3.plot(carbon_values[:24])
    ax3.set_xlabel("Hour")
    ax3.set_ylabel("gCO₂/kWh")
    ax3.set_title("Carbon Intensity Over Time")
    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(fig3)

    st.subheader("Carbon Emission per Job")
    fig4, ax4 = plt.subplots(figsize=(5, 3))
    ax4.bar(final_df.index, final_df["Carbon_Emission"])
    ax4.set_xlabel("Job Index")
    ax4.set_ylabel("Carbon Emission (gCO₂)")
    ax4.set_title("Carbon Emissions per Job")
    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(fig4)

    st.subheader("Workload Flexibility Distribution")
    counts = final_df["Flexible"].value_counts().sort_index()

    fig5, ax5 = plt.subplots(figsize=(5, 3))
    ax5.bar(["Non-Flexible", "Flexible"], counts)
    ax5.set_ylabel("Number of Jobs")
    ax5.set_title("Flexible vs Non-Flexible Workloads")
    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(fig5)

# ======================
# SECTION 6: GREEN SCHEDULING
# ======================

if section == "Green Scheduling Decisions":
    st.header("✅ Green Scheduling Decisions")

    st.subheader("Green Scheduling Score (Proposed)")
    st.latex(r"""
    Green\ Score =
    \alpha \left(\frac{1}{Energy}\right)
    +
    \beta \left(\frac{1}{Carbon\ Intensity}\right)
    +
    \gamma (Workload\ Flexibility)
    """)

    st.markdown("""
    - Lower energy → higher score  
    - Lower carbon intensity → higher score  
    - Flexible workload → higher score  

    Jobs with higher Green Score can be delayed to reduce emissions,
    while non-flexible jobs are executed immediately.
    """)

    # ======================
    # CARBON REDUCTION METRICS
    # ======================

    baseline_ci = final_df["Carbon_Intensity"].max()

    final_df["Baseline_Carbon_Emission"] = (
        final_df["Predicted_Energy"] * baseline_ci
    )

    total_baseline = final_df["Baseline_Carbon_Emission"].sum()
    total_ml = final_df["Carbon_Emission"].sum()

    carbon_saved = total_baseline - total_ml
    carbon_saved_pct = (carbon_saved / total_baseline) * 100

    st.subheader("🌍 Carbon Reduction Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Baseline Emissions (gCO₂)", f"{total_baseline:.2f}")
    col2.metric("ML-Based Emissions (gCO₂)", f"{total_ml:.2f}")
    col3.metric("Carbon Reduction (%)", f"{carbon_saved_pct:.2f}%")

    st.subheader("📊 Carbon Emissions: Baseline vs ML Scheduling")

    fig7, ax7 = plt.subplots(figsize=(5, 3))
    ax7.bar(["Baseline", "ML-Based"], [total_baseline, total_ml])
    ax7.set_ylabel("Total Carbon Emission (gCO₂)")
    ax7.set_title("Carbon Emission Reduction Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(fig7)

    
    st.subheader("Final Scheduling Output")
    st.dataframe(final_df)

    decision_counts = final_df["Scheduling_Decision"].value_counts()

    fig6, ax6 = plt.subplots(figsize=(5, 3))
    ax6.bar(decision_counts.index, decision_counts.values)
    ax6.set_ylabel("Number of Jobs")
    ax6.set_title("Scheduling Decisions")
    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(fig6)
