import numpy as np
import pandas as pd
import joblib
import pickle
import streamlit as st

# --- Page config ---
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Predictor")

# --- Load artifacts ---
try:
    model = joblib.load("model2.pkl")
except Exception as e:
    st.error(f"Could not load model2.pkl: {e}")
    st.stop()

try:
    le_obj = pickle.load(open("le.pkl", "rb"))
except Exception as e:
    st.error(f"Could not load le.pkl: {e}")
    st.stop()

try:
    scaler = joblib.load("sc2.pkl")
except Exception as e:
    st.error(f"Could not load sc2.pkl: {e}")
    st.stop()

# --- Helper utilities ---
def is_label_encoder(obj):
    return hasattr(obj, "classes_")

def safe_get_encoder(le_obj, key):
    """If le_obj is dict, return encoder for key; otherwise None."""
    if isinstance(le_obj, dict):
        # try common key names
        for k in [key, key.lower(), key.upper(), key.replace(" ", "_")]:
            if k in le_obj:
                return le_obj[k]
    return None

def encoder_knows(encoder, value):
    try:
        classes = getattr(encoder, "classes_", None)
        if classes is None:
            return False
        return str(value) in [str(c) for c in classes]
    except Exception:
        return False

def encode_with_le_or_fallback(colname, val):
    """
    Try to encode using an appropriate encoder from le_obj.
    If not available or value unseen, use fallback mapping.
    """
    # fallback maps (safe)
    edu_fallback = {"Graduate": 1, "Not Graduate": 0, "Other": 2}
    selfemp_fallback = {"Yes": 1, "No": 0}

    enc = safe_get_encoder(le_obj, colname)
    # If single encoder and it actually contains this feature's values, use it
    if enc is None and is_label_encoder(le_obj) and encoder_knows(le_obj, val):
        try:
            return int(le_obj.transform([val])[0])
        except Exception:
            pass

    # If dict encoder found and knows the label, use it
    if enc is not None and encoder_knows(enc, val):
        try:
            return int(enc.transform([val])[0])
        except Exception:
            pass

    # fallback mapping
    if colname == "education":
        return int(edu_fallback.get(val, 0))
    if colname == "self_employed":
        return int(selfemp_fallback.get(val, 0))
    # default
    return 0

def decode_target(pred):
    """Try to decode model's predicted label using le_obj when safe."""
    # If le_obj is dict with target encoder
    if isinstance(le_obj, dict):
        for key in ("loan_status", "target", "loanstatus", "Loan_Status"):
            enc = le_obj.get(key)
            if enc is not None and encoder_knows(enc, pred):
                try:
                    return enc.inverse_transform([pred])[0]
                except Exception:
                    pass
        # try any encoder that contains the class
        for enc in le_obj.values():
            if is_label_encoder(enc) and encoder_knows(enc, pred):
                try:
                    return enc.inverse_transform([pred])[0]
                except Exception:
                    pass
    else:
        # single encoder: decode only if it contains pred
        if is_label_encoder(le_obj) and encoder_knows(le_obj, pred):
            try:
                return le_obj.inverse_transform([pred])[0]
            except Exception:
                pass
    # fallback mapping
    try:
        return "Approved" if int(pred) == 1 else "Rejected"
    except Exception:
        return str(pred)

# --- Feature order (must match training) ---
FEATURE_ORDER = [
    'no_of_dependents',
    'education',
    'self_employed',
    'income_annum',
    'loan_amount',
    'loan_term',
    'cibil_score',
    'residential_assets_value',
    'commercial_assets_value',
    'luxury_assets_value',
    'bank_asset_value'
]

# --- Streamlit form ---
with st.form("loan_form"):
    st.subheader("Applicant & Loan Details")

    no_of_dependents = st.number_input("Number of Dependents", min_value=0, value=0, step=1)
    education = st.selectbox("Education", options=["Graduate", "Not Graduate", "Other"])
    self_employed = st.selectbox("Self-employed?", options=["Yes", "No"])
    income_annum = st.number_input("Annual Income (₹)", min_value=0, value=300000, step=10000)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=500000, step=10000)
    loan_term = st.number_input("Loan Term (months)", min_value=1, value=120, step=1)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700, step=1)

    st.subheader("Assets")
    residential_assets_value = st.number_input("Residential assets value (₹)", min_value=0, value=300000, step=10000)
    commercial_assets_value = st.number_input("Commercial assets value (₹)", min_value=0, value=0, step=10000)
    luxury_assets_value = st.number_input("Luxury assets value (₹)", min_value=0, value=0, step=10000)
    bank_asset_value = st.number_input("Bank / Liquid assets value (₹)", min_value=0, value=50000, step=10000)

    submitted = st.form_submit_button("🔍 Predict Loan Status")

# --- Prediction block ---
if submitted:
    try:
        # 1) Encode categorical features safely
        edu_enc = encode_with_le_or_fallback("education", education)
        selfemp_enc = encode_with_le_or_fallback("self_employed", self_employed)

        # 2) Build input dict in the expected FEATURE_ORDER
        raw = {
            'no_of_dependents': no_of_dependents,
            'education': edu_enc,
            'self_employed': selfemp_enc,
            'income_annum': income_annum,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'cibil_score': cibil_score,
            'residential_assets_value': residential_assets_value,
            'commercial_assets_value': commercial_assets_value,
            'luxury_assets_value': luxury_assets_value,
            'bank_asset_value': bank_asset_value
        }

        X_raw = pd.DataFrame([raw], columns=FEATURE_ORDER)

        # 3) Adapt to scaler expectations
        scaler_names = getattr(scaler, "feature_names_in_", None)
        scaler_n = getattr(scaler, "n_features_in_", None)

        if scaler_names is not None:
            expected_cols = [str(c) for c in scaler_names]
            X_expected = pd.DataFrame(columns=expected_cols)
            # put our known columns into expected frame
            for c in X_raw.columns:
                if c in X_expected.columns:
                    X_expected.loc[0, c] = X_raw.loc[0, c]
            X_expected = X_expected.fillna(0)
            X_for_scaler = X_expected[expected_cols]
        elif scaler_n is not None and scaler_n != X_raw.shape[1]:
            # pad with zeros if scaler expects more features, or trim if fewer
            if scaler_n > X_raw.shape[1]:
                pad = scaler_n - X_raw.shape[1]
                X_for_scaler = X_raw.copy()
                for i in range(pad):
                    X_for_scaler[f'pad_{i}'] = 0
                # ensure consistent column order
                X_for_scaler = X_for_scaler.reindex(columns=X_for_scaler.columns)
            elif scaler_n < X_raw.shape[1]:
                # trim to scaler_n columns (rare; trims rightmost)
                X_for_scaler = X_raw.iloc[:, :scaler_n]
            else:
                X_for_scaler = X_raw
        else:
            X_for_scaler = X_raw

        # Debugging info (optional, can remove in prod)
        st.write("Input raw shape:", X_raw.shape)
        st.write("Final shape for scaler:", X_for_scaler.shape)
        if scaler_names is not None:
            st.write("Scaler expected columns (sample):", list(scaler_names)[:12])

        # 4) Scale
        X_scaled = scaler.transform(X_for_scaler)
        # Ensure 2D numpy array for model
        X_scaled_arr = np.array(X_scaled)

        # 5) Predict
        pred_enc = model.predict(X_scaled_arr)[0]
        pred_label = decode_target(pred_enc)

        # 6) Show results
        if str(pred_label).lower() in ["approved", "yes", "1"]:
            st.success(f"✅ Loan Approved — {pred_label}")
        else:
            st.error(f"❌ Loan Not Approved — {pred_label}")

        # show probabilities if available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled_arr)[0]
            if len(probs) == 2:
                st.info(f"Approval probability: {probs[1]:.2f}")
            else:
                st.info("Class probabilities: " + ", ".join([f"{p:.2f}" for p in probs]))

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)