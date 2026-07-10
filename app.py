import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background:#f6f8fb;
}

h1,h2,h3{
    color:#0f172a;
}

[data-testid="metric-container"]{
    background:white;
    border-radius:15px;
    padding:18px;
    border:1px solid #e5e7eb;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load("student_performance_pipeline.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model:\n{e}")
    st.stop()

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def clean_columns(df):
    """Standardize column names"""
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_")
    )
    return df


def upload_dataset(key=None):
    """Upload CSV and return dataframe"""

    uploaded_file = st.file_uploader(
        " Upload CSV Dataset",
        type=["csv"],
        key=key
    )

    if uploaded_file is None:
        st.info("👆 Please upload a CSV file.")
        return None

    df = pd.read_csv(uploaded_file)
    df = clean_columns(df)

    return df


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title(" Student Analytics")

page = st.sidebar.radio(

    "Navigation",

    [

        " Dashboard",

        " Prediction",

        "ℹ About"

    ]

)

st.sidebar.markdown("---")

st.sidebar.success("Random Forest Model")

st.sidebar.info(
"""
Built with

• Python

• Streamlit

• Plotly

• Scikit-Learn
"""
)

# ==========================================================
# DASHBOARD PAGE
# ==========================================================

if page == " Dashboard":

    st.title(" Student Performance Dashboard")

    st.write(
        "Upload a dataset to explore interactive analytics."
    )

    df = upload_dataset("dashboard")

    if df is None:
        st.stop()

    # -----------------------------
    # Dataset Preview
    # -----------------------------

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    # -----------------------------
    # KPI Cards
    # -----------------------------

    numeric = df.select_dtypes(include="number")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", df.shape[0])

    c2.metric("Columns", df.shape[1])

    c3.metric("Numeric Columns", numeric.shape[1])

    c4.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.markdown("---")

    st.subheader("Summary Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    # ======================================================
    # FILTER DATA
    # ======================================================

    st.subheader(" Filter Dataset")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    selected_column = st.selectbox(
        "Select Numeric Column",
        numeric_cols
    )

    minimum = float(df[selected_column].min())
    maximum = float(df[selected_column].max())

    selected_range = st.slider(
        "Choose Range",
        minimum,
        maximum,
        (minimum, maximum)
    )

    filtered_df = df[
        (df[selected_column] >= selected_range[0]) &
        (df[selected_column] <= selected_range[1])
    ]

    st.success(f"Showing {len(filtered_df)} of {len(df)} rows")

    st.markdown("---")

    # ======================================================
    # HISTOGRAM
    # ======================================================

    st.subheader(" Distribution")

    hist_col = st.selectbox(
        "Histogram Feature",
        numeric_cols,
        key="hist"
    )

    fig = px.histogram(
        filtered_df,
        x=hist_col,
        nbins=30,
        title=f"{hist_col} Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # SCATTER PLOT
    # ======================================================

    st.markdown("---")

    st.subheader(" Scatter Plot")

    left, right = st.columns(2)

    with left:

        x_axis = st.selectbox(
            "X Axis",
            numeric_cols,
            key="scatter_x"
        )

    with right:

        y_axis = st.selectbox(
            "Y Axis",
            numeric_cols,
            index=1,
            key="scatter_y"
        )

    fig = px.scatter(
        filtered_df,
        x=x_axis,
        y=y_axis,
        color=y_axis,
        hover_data=filtered_df.columns
    )

    st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # BOXPLOT
    # ======================================================

    st.markdown("---")

    st.subheader(" Box Plot")

    box_feature = st.selectbox(
        "Select Feature",
        numeric_cols,
        key="box"
    )

    fig = px.box(
        filtered_df,
        y=box_feature,
        points="outliers"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # BAR CHART
    # ======================================================

    st.markdown("---")

    st.subheader(" Average Feature Values")

    mean_df = filtered_df[numeric_cols].mean().reset_index()

    mean_df.columns = ["Feature", "Average"]

    fig = px.bar(
        mean_df,
        x="Feature",
        y="Average",
        color="Average",
        text="Average"
    )

    fig.update_traces(texttemplate='%{text:.2f}')

    st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # PIE CHART
    # ======================================================

    categorical_cols = filtered_df.select_dtypes(include="object").columns.tolist()

    if len(categorical_cols) > 0:

        st.markdown("---")

        st.subheader(" Category Distribution")

        pie_column = st.selectbox(
            "Choose Category",
            categorical_cols
        )

        pie = filtered_df[pie_column].value_counts().reset_index()

        pie.columns = [pie_column, "Count"]

        fig = px.pie(
            pie,
            names=pie_column,
            values="Count",
            hole=0.45
        )

        st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # CORRELATION HEATMAP
    # ======================================================

    st.markdown("---")

    st.subheader(" Correlation Heatmap")

    corr = filtered_df[numeric_cols].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # TOP STUDENTS
    # ======================================================

    if "performance_index" in filtered_df.columns:

        st.markdown("---")

        st.subheader(" Top 10 Students")

        top = filtered_df.sort_values(
            "performance_index",
            ascending=False
        ).head(10)

        st.dataframe(
            top,
            use_container_width=True
        )

    # ======================================================
    # AUTOMATIC INSIGHTS
    # ======================================================

    st.markdown("---")

    st.subheader(" Insights")

    highest = filtered_df[numeric_cols].mean().idxmax()
    lowest = filtered_df[numeric_cols].mean().idxmin()

    col1, col2 = st.columns(2)

    col1.success(f"Highest Average Feature\n\n{highest}")

    col2.warning(f"Lowest Average Feature\n\n{lowest}")

    if "performance_index" in filtered_df.columns:

        correlation = (
            filtered_df[numeric_cols]
            .corr()["performance_index"]
            .drop("performance_index")
            .abs()
            .sort_values(ascending=False)
        )

        st.info(
            f"Most Important Feature: **{correlation.index[0]}**"
        )

    # ======================================================
    # DOWNLOAD DATA
    # ======================================================

    st.markdown("---")

    csv = filtered_df.to_csv(index=False).encode()

    st.download_button(

        " Download Filtered Dataset",

        csv,

        "filtered_dataset.csv",

        "text/csv"

    )



# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == " Prediction":

    st.title(" Student Performance Prediction")

    st.write(
        "Upload a CSV file to predict the Performance Index for each student."
    )

    uploaded_prediction = st.file_uploader(
        "Upload CSV for Prediction",
        type=["csv"],
        key="prediction_file"
    )

    if uploaded_prediction is not None:

        predict_df = pd.read_csv(uploaded_prediction)

        predict_df.columns = (
            predict_df.columns
            .str.lower()
            .str.strip()
            .str.replace(" ", "_")
            .str.replace("/", "_")
        )

        st.success("Dataset uploaded successfully!")

        st.subheader("Preview")

        st.dataframe(
            predict_df.head(),
            use_container_width=True
        )

        # Make Predictions
        predictions = model.predict(predict_df)

        predict_df["Predicted Performance Index"] = predictions.round(2)

        st.success("Prediction Completed Successfully!")

        st.subheader("Prediction Results")

        st.dataframe(
            predict_df,
            use_container_width=True
        )

        st.metric(
            "Average Predicted Score",
            round(predictions.mean(),2)
        )

        # ============================
        # Prediction Distribution
        # ============================

        st.markdown("---")

        st.subheader(" Prediction Distribution")

        fig = px.histogram(
            predict_df,
            x="Predicted Performance Index",
            nbins=25,
            color_discrete_sequence=["royalblue"]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ============================
        # Top Students
        # ============================

        st.markdown("---")

        st.subheader("🏆 Top Predicted Students")

        top10 = predict_df.sort_values(
            by="Predicted Performance Index",
            ascending=False
        ).head(10)

        st.dataframe(
            top10,
            use_container_width=True
        )

        
        # ============================================
        # DOWNLOAD PREDICTIONS
        # ============================================

        st.markdown("---")
        st.subheader(" Download Predictions")

        csv = predict_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label=" Download Predictions CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
   
# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "ℹ About":

    st.title(" Student Performance Analytics Dashboard")

    st.markdown("""
    ## Project Overview

    The **Student Performance Analytics Dashboard** is an interactive web application built using **Python**, **Streamlit**, and **Machine Learning**.

    The application allows users to upload their own student dataset, explore it through interactive visualizations, and generate performance predictions using a trained **Random Forest Regression** model.

    ---

    ##  Key Features

    -  Upload your own CSV dataset
    - Interactive dashboard and visualizations
    - Histogram, Scatter Plot, Box Plot & Bar Chart
    - Pie Chart for categorical analysis
    - Correlation Heatmap
    -  Automatic dataset insights
    - Student Performance Prediction
    - Download prediction results as CSV

    ---

    ##  Machine Learning Model

    **Algorithm Used**

    - Random Forest Regressor

    The model predicts the **Performance Index** of students based on academic and study-related features.

    ---

    ## 🛠 Technologies Used

    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Plotly
    - Scikit-Learn
    - Joblib

    ---

    ##  Purpose

    This project demonstrates practical skills in:

    - Data Cleaning
    - Exploratory Data Analysis (EDA)
    - Data Visualization
    - Machine Learning
    - Model Deployment
    - Interactive Dashboard Development

    ---

    ##  Developer

    **Meerab Fatima**

    BS Data Science Student

    Portfolio Project | 2026
        """)

    st.markdown("---")

    st.caption(
            "Student Performance Analytics Dashboard • Built with Streamlit & Scikit-Learn"
        )
                          
        
    