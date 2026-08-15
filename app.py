import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------
# CreditWise - Loan Approval Prediction
# Based on the workflow in creditwise_loan_system.ipynb
# ---------------------------------------------------------

st.set_page_config(
    page_title="CreditWise Loan Approval",
    page_icon="🏦",
    layout="wide"
)

DATA_FILE = "loan_approval_data.csv"

OHE_COLS = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category",
]

NUMERIC_COLS = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Age",
    "Dependents",
    "Credit_Score",
    "Existing_Loans",
    "DTI_Ratio",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Loan_Term",
]


@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_FILE)

    # Same basic cleaning approach used in the notebook:
    # numerical missing values -> mean
    # categorical missing values -> most frequent
    numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    num_imp = SimpleImputer(strategy="mean")
    df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

    # Remove Applicant_ID, as done in the notebook
    df = df.drop(columns=["Applicant_ID"], errors="ignore")

    # Target encoding: No -> 0, Yes -> 1
    target_encoder = LabelEncoder()
    df["Loan_Approved"] = target_encoder.fit_transform(df["Loan_Approved"])

    # Education_Level was LabelEncoded in the notebook
    education_encoder = LabelEncoder()
    df["Education_Level"] = education_encoder.fit_transform(df["Education_Level"])

    # One-hot encode the six categorical feature columns
    ohe = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore"
    )

    encoded = ohe.fit_transform(df[OHE_COLS])
    encoded_cols = ohe.get_feature_names_out(OHE_COLS)

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoded_cols,
        index=df.index
    )

    df = pd.concat(
        [
            df.drop(columns=OHE_COLS),
            encoded_df
        ],
        axis=1
    )

    X = df.drop(columns=["Loan_Approved"])
    y = df["Loan_Approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression used in the notebook
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    accuracy = model.score(X_test_scaled, y_test)

    return {
        "model": model,
        "scaler": scaler,
        "num_imputer": num_imp,
        "cat_imputer": cat_imp,
        "education_encoder": education_encoder,
        "ohe": ohe,
        "feature_columns": X.columns.tolist(),
        "accuracy": accuracy,
        "target_encoder": target_encoder,
        "raw_df": pd.read_csv(DATA_FILE),
    }


def preprocess_input(input_df, artifacts):
    # Apply the same preprocessing learned during training
    df = input_df.copy()

    df["Applicant_ID"] = 0.0

    # Impute numeric columns
    numeric_for_imputation = [
        "Applicant_ID",
        "Applicant_Income",
        "Coapplicant_Income",
        "Age",
        "Dependents",
        "Credit_Score",
        "Existing_Loans",
        "DTI_Ratio",
        "Savings",
        "Collateral_Value",
        "Loan_Amount",
        "Loan_Term",
    ]
    df[numeric_for_imputation] = artifacts["num_imputer"].transform(
        df[numeric_for_imputation]
    )

    # Impute categorical columns
    df[OHE_COLS + ["Education_Level"]] = artifacts["cat_imputer"].transform(
        df[OHE_COLS + ["Education_Level"]]
    )

    # Education label encoding
    df["Education_Level"] = artifacts["education_encoder"].transform(
        df["Education_Level"]
    )

    # One-hot encode
    encoded = artifacts["ohe"].transform(df[OHE_COLS])
    encoded_df = pd.DataFrame(
        encoded,
        columns=artifacts["ohe"].get_feature_names_out(OHE_COLS),
        index=df.index,
    )

    df = pd.concat(
        [
            df.drop(columns=OHE_COLS),
            encoded_df
        ],
        axis=1
    )

    # Match exact training feature order
    df = df[artifacts["feature_columns"]]

    return artifacts["scaler"].transform(df)


# ---------------------------------------------------------
# Load / train
# ---------------------------------------------------------
st.title("🏦 CreditWise Loan Approval System")
st.write(
    "Enter applicant information below to estimate whether the loan "
    "application is likely to be approved."
)

try:
    artifacts = train_model()
except FileNotFoundError:
    st.error(
        f"'{DATA_FILE}' was not found. Put loan_approval_data.csv "
        "in the same folder as app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not train the model: {e}")
    st.stop()


with st.sidebar:
    st.header("Model Information")
    st.metric("Test Accuracy", f"{artifacts['accuracy'] * 100:.2f}%")
    st.info(
        "Model: Logistic Regression\n\n"
        "The model is trained from loan_approval_data.csv "
        "when the Streamlit app starts."
    )


df_raw = artifacts["raw_df"]

# Helper functions for sensible defaults
def median_value(column, fallback=0.0):
    value = pd.to_numeric(df_raw[column], errors="coerce").median()
    return fallback if pd.isna(value) else float(value)


def mode_value(column, fallback):
    values = df_raw[column].dropna()
    return values.mode().iloc[0] if not values.empty else fallback


# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
with st.form("loan_form"):
    st.subheader("Applicant Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0.0,
            value=median_value("Applicant_Income"),
            step=100.0,
        )

        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0.0,
            value=median_value("Coapplicant_Income"),
            step=100.0,
        )

        age = st.number_input(
            "Age",
            min_value=18.0,
            max_value=100.0,
            value=max(18.0, min(100.0, median_value("Age", 30))),
            step=1.0,
        )

        dependents = st.number_input(
            "Dependents",
            min_value=0.0,
            max_value=20.0,
            value=max(0.0, median_value("Dependents")),
            step=1.0,
        )

    with col2:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300.0,
            max_value=900.0,
            value=max(300.0, min(900.0, median_value("Credit_Score", 650))),
            step=1.0,
        )

        existing_loans = st.number_input(
            "Existing Loans",
            min_value=0.0,
            value=max(0.0, median_value("Existing_Loans")),
            step=1.0,
        )

        dti_ratio = st.number_input(
            "DTI Ratio",
            min_value=0.0,
            max_value=2.0,
            value=max(0.0, min(2.0, median_value("DTI_Ratio", 0.30))),
            step=0.01,
        )

        savings = st.number_input(
            "Savings",
            min_value=0.0,
            value=median_value("Savings"),
            step=100.0,
        )

    with col3:
        collateral_value = st.number_input(
            "Collateral Value",
            min_value=0.0,
            value=median_value("Collateral_Value"),
            step=100.0,
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=median_value("Loan_Amount"),
            step=100.0,
        )

        loan_term = st.number_input(
            "Loan Term (months)",
            min_value=1.0,
            max_value=600.0,
            value=max(1.0, min(600.0, median_value("Loan_Term", 60))),
            step=1.0,
        )

        employment_options = sorted(
            df_raw["Employment_Status"].dropna().astype(str).unique().tolist()
        )
        employment_status = st.selectbox(
            "Employment Status",
            employment_options,
            index=employment_options.index(
                mode_value("Employment_Status", employment_options[0])
            ),
        )

    st.subheader("Application Information")

    col4, col5, col6 = st.columns(3)

    with col4:
        marital_options = sorted(
            df_raw["Marital_Status"].dropna().astype(str).unique().tolist()
        )
        marital_status = st.selectbox(
            "Marital Status",
            marital_options,
            index=marital_options.index(
                mode_value("Marital_Status", marital_options[0])
            ),
        )

        purpose_options = sorted(
            df_raw["Loan_Purpose"].dropna().astype(str).unique().tolist()
        )
        loan_purpose = st.selectbox(
            "Loan Purpose",
            purpose_options,
            index=purpose_options.index(
                mode_value("Loan_Purpose", purpose_options[0])
            ),
        )

    with col5:
        property_options = sorted(
            df_raw["Property_Area"].dropna().astype(str).unique().tolist()
        )
        property_area = st.selectbox(
            "Property Area",
            property_options,
            index=property_options.index(
                mode_value("Property_Area", property_options[0])
            ),
        )

        education_options = sorted(
            df_raw["Education_Level"].dropna().astype(str).unique().tolist()
        )
        education_level = st.selectbox(
            "Education Level",
            education_options,
            index=education_options.index(
                mode_value("Education_Level", education_options[0])
            ),
        )

    with col6:
        gender_options = sorted(
            df_raw["Gender"].dropna().astype(str).unique().tolist()
        )
        gender = st.selectbox(
            "Gender",
            gender_options,
            index=gender_options.index(
                mode_value("Gender", gender_options[0])
            ),
        )

        employer_options = sorted(
            df_raw["Employer_Category"].dropna().astype(str).unique().tolist()
        )
        employer_category = st.selectbox(
            "Employer Category",
            employer_options,
            index=employer_options.index(
                mode_value("Employer_Category", employer_options[0])
            ),
        )

    submitted = st.form_submit_button(
        "🔍 Predict Loan Approval",
        use_container_width=True,
    )


if submitted:
    input_data = pd.DataFrame(
        [
            {
                "Applicant_Income": applicant_income,
                "Coapplicant_Income": coapplicant_income,
                "Employment_Status": employment_status,
                "Age": age,
                "Marital_Status": marital_status,
                "Dependents": dependents,
                "Credit_Score": credit_score,
                "Existing_Loans": existing_loans,
                "DTI_Ratio": dti_ratio,
                "Savings": savings,
                "Collateral_Value": collateral_value,
                "Loan_Amount": loan_amount,
                "Loan_Term": loan_term,
                "Loan_Purpose": loan_purpose,
                "Property_Area": property_area,
                "Education_Level": education_level,
                "Gender": gender,
                "Employer_Category": employer_category,
            }
        ]
    )

    try:
        X_input = preprocess_input(input_data, artifacts)

        prediction = artifacts["model"].predict(X_input)[0]
        probability = artifacts["model"].predict_proba(X_input)[0]

        # target encoder preserves the notebook's LabelEncoder target mapping
        predicted_label = artifacts["target_encoder"].inverse_transform([prediction])[0]

        st.divider()
        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if predicted_label == "Yes":
                st.success("### ✅ Loan Likely Approved")
            else:
                st.error("### ❌ Loan Likely Not Approved")

        with result_col2:
            st.metric(
                "Approval Probability",
                f"{probability[prediction] * 100:.2f}%"
            )

        st.progress(float(probability[1]) if len(probability) > 1 else 0.0)

        st.caption(
            "This is a machine-learning prediction based on the training "
            "dataset, not a real banking/financial approval decision."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
