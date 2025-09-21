import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Smart Grid Analyzer", layout="wide")
st.title("⚡ Smart Grid Analyzer")
st.write("Upload a dataset and explore clustering + regression for renewable energy distribution.")

# File uploader
uploaded_file = st.file_uploader("energy_data.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    # -------------------------
    # Clustering Section
    # -------------------------
    st.subheader("🔎 Clustering Analysis")
    features = st.multiselect("Select features for clustering", df.columns.tolist(), default=df.columns[:3].tolist())

    if len(features) >= 2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])

        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        st.write("Cluster counts:")
        st.bar_chart(df['Cluster'].value_counts())

        # Plot clusters
        fig, ax = plt.subplots()
        ax.scatter(df[features[0]], df[features[1]], c=df['Cluster'], cmap="viridis")
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        st.pyplot(fig)
    else:
        st.warning(" Please select at least 2 features for clustering.")

    # -------------------------
    # Regression Section
    # -------------------------
    st.subheader("📈 Regression Prediction")
    target = st.selectbox("Select target column (y)", df.columns.tolist())
    predictors = st.multiselect("Select predictors (X)", [col for col in df.columns if col != target])

    if predictors:
        X = df[predictors]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        results = pd.DataFrame({"Actual": y_test.values, "Predicted": predictions})

        st.write("Prediction Results")
        st.dataframe(results.head())

        # Plot predictions
        fig2, ax2 = plt.subplots()
        ax2.plot(y_test.values, label="Actual", color="blue")
        ax2.plot(predictions, label="Predicted", color="red")
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.info("Please select predictors to run regression.")
else:
    st.info("⬆️ Please upload a CSV dataset to begin.")

print("By Ronak Baniabbasi")
