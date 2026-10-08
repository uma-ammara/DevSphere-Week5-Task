"""Phishing URL Detector - Streamlit app. Loads the model trained in Google Colab (no training here)."""

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import streamlit as st

st.set_page_config(page_title="Phishing URL Detector", page_icon="🔐", layout="wide")
sns.set_theme(style="whitegrid")
st.markdown(
    """
    <style>
        #MainMenu, footer {visibility: hidden;}
        .block-container {padding-top: 2rem; max-width: 1100px;}
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_DIR = Path(__file__).parent / "model_files"
REQUIRED_FILES = ["phishing_model.pkl", "feature_names.pkl", "model_config.json", "feature_importance.csv"]

LOW_RISK_BELOW = 0.40    # phishing probability thresholds for the risk level
HIGH_RISK_FROM = 0.70
SENSITIVE_WORDS = [
    "login", "signin", "secure", "account", "update", "verify", "bank", "confirm",
    "password", "paypal", "webscr", "billing", "wallet", "suspend", "security",
]

st.title("🔐 Phishing URL Detector")
st.caption("Random Forest model trained in Google Colab. Enter a URL to check it.")

# ----------------------------------------------------------------------------
# Load the files exported from Colab
# ----------------------------------------------------------------------------
missing = [f for f in REQUIRED_FILES if not (MODEL_DIR / f).exists()]
if missing:
    st.error(f"Missing file(s) in `model_files/`: {missing}")
    st.info("Put the files from `phishing_model_files.zip` (Colab Cell 16) inside a `model_files` folder next to app.py.")
    st.stop()


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model = joblib.load(MODEL_DIR / "phishing_model.pkl")
    features = joblib.load(MODEL_DIR / "feature_names.pkl")
    with open(MODEL_DIR / "model_config.json", encoding="utf-8") as f:
        config = json.load(f)
    importance = pd.read_csv(MODEL_DIR / "feature_importance.csv")
    return model, features, config, importance


try:
    model, FEATURES, config, imp_df = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.info("Install the scikit-learn version used in Colab: `pip install scikit-learn==1.6.1`")
    st.stop()

METRICS = config["metrics_test_set"]
if sklearn.__version__ != config.get("sklearn_version"):
    st.warning(f"scikit-learn is {sklearn.__version__} but the model was trained with "
               f"{config.get('sklearn_version')}. Install the same version to avoid problems.")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def validate_url(raw):
    """Return (url, error). error is None when the URL is acceptable."""
    url = (raw or "").strip()
    if not url:
        return None, "Please enter a URL."
    if re.search(r"\s", url):
        return None, "A URL cannot contain spaces."
    if len(url) > 2048:
        return None, "The URL is too long (maximum 2048 characters)."
    has_scheme = re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url) is not None
    if has_scheme and not url.lower().startswith(("http://", "https://")):
        return None, "Only http:// and https:// URLs are supported."
    try:
        host = urlparse(url if has_scheme else "http://" + url).hostname
    except ValueError:
        return None, "This does not look like a valid URL."
    if not host:
        return None, "No domain name found in the URL."
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host) is None:
        labels = host.split(".")
        if len(labels) < 2 or any(not lab for lab in labels):
            return None, "The domain looks incomplete (example: example.com)."
        for lab in labels:
            if len(lab) > 63 or lab.startswith("-") or lab.endswith("-") or \
                    not all(ch.isalnum() or ch == "-" for ch in lab):
                return None, f"Invalid characters in the domain part '{lab}'."
        if len(labels[-1]) < 2:
            return None, "The domain ending (TLD) is too short."
    return url, None


def extract_features(url, valid_url, nb_or):
    """Approximate feature extraction (the dataset does not document how its columns were made)."""
    low = url.lower()
    has_scheme = re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url) is not None
    parsed = urlparse(url if has_scheme else "http://" + url)
    return {
        "url_length": len(url),
        "valid_url": int(valid_url),
        "at_symbol": url.count("@"),
        "sensitive_words_count": sum(1 for w in SENSITIVE_WORDS if w in low),
        "path_length": len(parsed.path),
        "isHttps": 1 if low.startswith("https://") else 0,
        "nb_dots": url.count("."),
        "nb_hyphens": url.count("-"),
        "nb_and": url.count("&"),
        "nb_or": int(nb_or),
        "nb_www": low.count("www"),
        "nb_com": low.count("com"),
        "nb_underscore": url.count("_"),
    }


def predict(values):
    row = pd.DataFrame([[float(values[f]) for f in FEATURES]], columns=FEATURES)
    proba = model.predict_proba(row)[0]
    classes = list(model.classes_)
    return int(model.predict(row)[0]), float(proba[classes.index(0)]), float(proba[classes.index(1)])


def risk_level(p_phish):
    if p_phish < LOW_RISK_BELOW:
        return "🟢 LOW"
    if p_phish < HIGH_RISK_FROM:
        return "🟡 MEDIUM"
    return "🔴 HIGH"


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_check, tab_perf, tab_imp = st.tabs(["🔎 Check URL", "📊 Model performance", "🌲 Feature importance"])

with tab_check:
    with st.form("check_form", border=False):
        url_input = st.text_input("Website URL", placeholder="https://example.com/login")
        with st.expander("Advanced: two dataset features that cannot be computed from a URL"):
            st.caption("The dataset does not document how `valid_url` and `nb_or` were produced, "
                       "so you can set them here. Defaults (0) are the most common values.")
            c1, c2 = st.columns(2)
            valid_url = c1.selectbox("valid_url", [0, 1], index=0)
            nb_or = c2.number_input("nb_or", min_value=0, max_value=50, value=0, step=1)
        submitted = st.form_submit_button("Check URL", type="primary")

    if submitted:
        url, error = validate_url(url_input)
        if error:
            st.warning(error)
        else:
            values = extract_features(url, valid_url, nb_or)
            label, p_phish, p_legit = predict(values)

            with st.container(border=True):
                if label == 0:
                    st.error("## 🚨 PHISHING URL")
                else:
                    st.success("## ✅ LEGITIMATE URL")
                m1, m2, m3 = st.columns(3)
                m1.metric("Phishing probability", f"{p_phish * 100:.1f}%")
                m2.metric("Legitimate probability", f"{p_legit * 100:.1f}%")
                m3.metric("Risk level", risk_level(p_phish))
                st.progress(min(max(p_phish, 0.0), 1.0))
                st.caption(f"Risk: LOW below {LOW_RISK_BELOW:.0%}, MEDIUM up to {HIGH_RISK_FROM:.0%}, "
                           f"HIGH above. Model test accuracy is about {METRICS['accuracy'] * 100:.0f}%, "
                           "so use this as a signal, not a final verdict.")

            with st.expander("Extracted URL features", expanded=True):
                notes = {"valid_url": "set in Advanced", "nb_or": "set in Advanced"}
                st.dataframe(
                    pd.DataFrame({"Feature": FEATURES,
                                  "Value": [values[f] for f in FEATURES],
                                  "Note": [notes.get(f, "") for f in FEATURES]}),
                    hide_index=True, width="stretch")

with tab_perf:
    st.caption(f"Evaluated on {config['data_summary']['test_rows']} test URLs the model never saw during training.")
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{METRICS['accuracy'] * 100:.1f}%")
    cols[1].metric("Precision", f"{METRICS['precision_phishing'] * 100:.1f}%")
    cols[2].metric("Recall", f"{METRICS['recall_phishing'] * 100:.1f}%")
    cols[3].metric("F1-score", f"{METRICS['f1_phishing'] * 100:.1f}%")
    cols[4].metric("ROC-AUC", f"{METRICS['roc_auc']:.3f}")
    st.caption("Precision, recall and F1 are for the phishing class.")

    left, right = st.columns(2)
    with left:
        # The key name differs between notebook versions, so find it by prefix
        cm_key = next((k for k in METRICS if k.startswith("confusion_matrix")), None)
        if cm_key is None:
            st.warning("Confusion matrix not found in model_config.json.")
        else:
            cm = np.array(METRICS[cm_key])
            fig, ax = plt.subplots(figsize=(4.6, 3.8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 15},
                        xticklabels=["Phishing", "Legitimate"], yticklabels=["Phishing", "Legitimate"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion matrix")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        ax.plot(config["roc_curve"]["fpr"], config["roc_curve"]["tpr"], color="tab:blue", linewidth=2.5,
                label=f"AUC = {METRICS['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve")
        ax.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    o = config["overfitting_check"]
    st.info(f"Training accuracy is {o['train_accuracy'] * 100:.1f}% but test accuracy is "
            f"{o['test_accuracy'] * 100:.1f}%, so the model partly memorised its training data. "
            "The test score is the realistic one.")

with tab_imp:
    d = imp_df.sort_values("importance")
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(d["feature"], d["importance"] * 100, color="steelblue")
    ax.set_xlabel("Importance (%)")
    ax.set_title("Which features the Random Forest relies on")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    top = imp_df.sort_values("importance", ascending=False).iloc[0]
    st.caption(f"Most important feature: `{top['feature']}`.")
