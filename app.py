import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

st.set_page_config(page_title="AutoML Trainer", layout="wide")
st.title("🚀 Smart AutoML Streamlit App")

# =========================
# Helper Functions
# =========================

def detect_problem_type(y):
    if y.dtype == "object" or y.nunique() <= 10:
        return "classification"
    return "regression"

def remove_outliers(df, numeric_cols):
    before = len(df)
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df, before - len(df)

# =========================
# Upload Dataset
# =========================

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Dataset Preview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)

    # =========================
    # Dataset Diagnostics
    # =========================

    st.subheader("🛑 Dataset Issues (Before Cleaning)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Missing Values", df.isnull().sum().sum())
    col2.metric("Duplicate Rows", df.duplicated().sum())
    col3.metric("Total Columns", df.shape[1])

    st.text("Dataset Info:")
    st.text(df.info())

    # =========================
    # Target Selection
    # =========================

    target = st.selectbox("🎯 Select Target Column", df.columns)

    if target:
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

        # =========================
        # Cleaning Button
        # =========================

        if st.button("🧹 Clean Dataset"):
            df_clean = df.copy()

            # Remove duplicates
            df_clean.drop_duplicates(inplace=True)

            # Remove missing values
            df_clean.dropna(inplace=True)

            # Remove outliers
            df_clean, outliers_removed = remove_outliers(
                df_clean, [c for c in num_cols if c != target]
            )

            st.success("Dataset cleaned successfully!")

            st.subheader("✅ Dataset Info After Cleaning")
            st.write("Shape:", df_clean.shape)
            st.text(df_clean.info())

            # =========================
            # Visualization
            # =========================

            st.subheader("📈 Data Visualizations")

            if df_clean[target].dtype != "object":
                fig, ax = plt.subplots()
                sns.histplot(df_clean[target], kde=True, ax=ax)
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots()
                df_clean[target].value_counts().plot(kind="bar", ax=ax)
                st.pyplot(fig)

            # Correlation heatmap
            if len(num_cols) > 1:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.heatmap(df_clean[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)

            # =========================
            # Model Training
            # =========================

            X = df_clean.drop(columns=[target])
            y = df_clean[target]

            problem_type = detect_problem_type(y)
            st.info(f"Detected Problem Type: **{problem_type.upper()}**")

            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer([
                ("num", numeric_pipeline, X.select_dtypes(include=["int64","float64"]).columns),
                ("cat", categorical_pipeline, X.select_dtypes(include=["object"]).columns)
            ])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if problem_type == "classification":
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest": RandomForestClassifier(),
                    "SVM": SVC()
                }
            else:
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(),
                    "SVR": SVR()
                }

            best_model = None
            best_score = -np.inf
            best_model_name = ""

            results = []

            for name, model in models.items():
                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", model)
                ])

                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)

                score = accuracy_score(y_test, preds) if problem_type == "classification" else r2_score(y_test, preds)

                results.append([name, score])

                if score > best_score:
                    best_score = score
                    best_model = pipeline
                    best_model_name = name

            results_df = pd.DataFrame(
                results,
                columns=["Model", "Accuracy" if problem_type=="classification" else "R2 Score"]
            )

            st.subheader("🏆 Model Comparison")
            st.dataframe(results_df)

            st.success(f"✅ Best Model Selected: **{best_model_name}**")

            # =========================
            # Downloads
            # =========================

            joblib.dump(best_model, "best_model.pkl")

            st.download_button(
                "⬇️ Download Best Model (.pkl)",
                data=open("best_model.pkl", "rb"),
                file_name="best_model.pkl"
            )

            st.download_button(
                "⬇️ Download Cleaned Dataset (.csv)",
                data=df_clean.to_csv(index=False),
                file_name="cleaned_dataset.csv"
            )

            # =========================
            # Prediction
            # =========================

            st.subheader("🔮 Make Prediction")

            input_data = {}
            for col in X.columns:
                if col in X.select_dtypes(include=["int64","float64"]).columns:
                    input_data[col] = st.number_input(col, float(X[col].mean()))
                else:
                    input_data[col] = st.text_input(col)

            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])
                prediction = best_model.predict(input_df)
                st.success(f"Prediction Result: {prediction[0]}")
