"""
Phishing URL Detector - Streamlit app
"""

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

MODEL_DIR = Path(__file__).parent / "model_files"
REQUIRED_FILES = [
    "phishing_model.pkl",
    "feature_names.pkl",
    "model_config.json",
    "feature_importance.csv",
    "permutation_importance.csv",
]

# Risk thresholds on the model's PHISHING probability
LOW_RISK_BELOW = 0.40
HIGH_RISK_FROM = 0.70

# Words counted for `sensitive_words_count` in URL mode (approximation, see About tab)
SENSITIVE_WORDS = [
    "login", "signin", "secure", "account", "update", "verify", "bank", "confirm",
    "password", "paypal", "webscr", "billing", "wallet", "suspend", "security",
]

FEATURE_HELP = {
    "url_length": "Total number of characters in the URL.",
    "valid_url": "Dataset flag (0/1). Its exact definition is not documented in the file.",
    "at_symbol": "Number of '@' characters.",
    "sensitive_words_count": "Number of suspicious words (login, secure, verify, ...).",
    "path_length": "Length of the path part of the URL.",
    "isHttps": "1 if the URL starts with https://, else 0.",
    "nb_dots": "Number of '.' characters.",
    "nb_hyphens": "Number of '-' characters.",
    "nb_and": "Number of '&' characters.",
    "nb_or": "Dataset count feature. Its exact definition is not documented in the file.",
    "nb_www": "Number of times 'www' appears.",
    "nb_com": "Number of times 'com' appears.",
    "nb_underscore": "Number of '_' characters.",
}


# ----------------------------------------------------------------------------
# Loading the files exported from Colab
# ----------------------------------------------------------------------------
missing = [f for f in REQUIRED_FILES if not (MODEL_DIR / f).exists()]
if missing:
    st.title("🔐 Phishing URL Detector")
    st.error(f"Missing file(s) in `{MODEL_DIR}`: {missing}")
    st.info(
        "Download `phishing_model_files.zip` from Google Colab (Cell 16), unzip it, and put the "
        "5 files inside a folder named `model_files` next to `app.py`."
    )
    st.stop()


@st.cache_resource(show_spinner="Loading trained model...")
def load_artifacts():
    model = joblib.load(MODEL_DIR / "phishing_model.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    with open(MODEL_DIR / "model_config.json", encoding="utf-8") as f:
        config = json.load(f)
    importance = pd.read_csv(MODEL_DIR / "feature_importance.csv")
    permutation = pd.read_csv(MODEL_DIR / "permutation_importance.csv")
    return model, feature_names, config, importance, permutation


try:
    model, FEATURES, config, imp_df, perm_df = load_artifacts()
except Exception as exc:  # corrupted file or incompatible scikit-learn version
    st.title("🔐 Phishing URL Detector")
    st.error(f"Could not load the model: {exc}")
    st.info(
        "This usually means a different scikit-learn version. Install the version used in Colab: "
        "`pip install scikit-learn==1.6.1` (check `sklearn_version` in model_config.json)."
    )
    st.stop()

METRICS = config["metrics_test_set"]
MEDIANS = config["feature_medians_by_class"]
RANGES = config["feature_ranges_in_training_data"]
BINARY = set(config.get("binary_features", []))


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def validate_url(raw):
    """Return (clean_url, error_message). error_message is None when the URL is acceptable."""
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
        parsed = urlparse(url if has_scheme else "http://" + url)
        host = parsed.hostname
    except ValueError:
        return None, "This does not look like a valid URL."
    if not host:
        return None, "No domain name found in the URL."

    is_ip = re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host) is not None
    if not is_ip:
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


def extract_url_features(url):
    """
    APPROXIMATE feature extraction. The dataset does not document exactly how its columns were
    computed, so these are reasonable counts. valid_url and nb_or are NOT computed here.
    """
    low = url.lower()
    has_scheme = re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url) is not None
    parsed = urlparse(url if has_scheme else "http://" + url)
    return {
        "url_length": len(url),
        "at_symbol": url.count("@"),
        "sensitive_words_count": sum(1 for w in SENSITIVE_WORDS if w in low),
        "path_length": len(parsed.path),
        "isHttps": 1 if low.startswith("https://") else 0,
        "nb_dots": url.count("."),
        "nb_hyphens": url.count("-"),
        "nb_and": url.count("&"),
        "nb_www": low.count("www"),
        "nb_com": low.count("com"),
        "nb_underscore": url.count("_"),
    }


def predict_features(values):
    """Run the trained model on a dict of the 13 feature values (columns in training order)."""
    row = pd.DataFrame([[float(values[f]) for f in FEATURES]], columns=FEATURES)
    proba = model.predict_proba(row)[0]
    classes = list(model.classes_)
    p_phish = float(proba[classes.index(0)])
    p_legit = float(proba[classes.index(1)])
    label_class = int(model.predict(row)[0])
    return label_class, p_phish, p_legit


def risk_level(p_phish):
    if p_phish < LOW_RISK_BELOW:
        return "LOW", "🟢"
    if p_phish < HIGH_RISK_FROM:
        return "MEDIUM", "🟡"
    return "HIGH", "🔴"


def show_result(values, source):
    label_class, p_phish, p_legit = predict_features(values)
    level, icon = risk_level(p_phish)

    if label_class == 0:
        st.error("## 🚨 PHISHING URL")
    else:
        st.success("## ✅ LEGITIMATE URL")

    c1, c2, c3 = st.columns(3)
    c1.metric("Phishing probability", f"{p_phish * 100:.1f}%")
    c2.metric("Legitimate probability", f"{p_legit * 100:.1f}%")
    c3.metric("Risk level", f"{icon} {level}")
    st.progress(min(max(p_phish, 0.0), 1.0), text=f"Phishing probability: {p_phish * 100:.1f}%")
    st.caption(
        f"Risk level is based on phishing probability: LOW < {LOW_RISK_BELOW:.0%}, "
        f"MEDIUM {LOW_RISK_BELOW:.0%}-{HIGH_RISK_FROM:.0%}, HIGH >= {HIGH_RISK_FROM:.0%}. "
        f"The model's test accuracy is {METRICS['accuracy'] * 100:.1f}%, so treat this as a signal, "
        "not a verdict."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Probability breakdown**")
        fig, ax = plt.subplots(figsize=(5, 2.2))
        ax.barh(["Legitimate", "Phishing"], [p_legit * 100, p_phish * 100],
                color=["#2ca02c", "#d62728"])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)")
        for i, v in enumerate([p_legit * 100, p_phish * 100]):
            ax.text(min(v + 1, 88), i, f"{v:.1f}%", va="center")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Features sent to the model** (vs typical values in the training data)")
        table = pd.DataFrame({
            "Feature": FEATURES,
            "Your value": [values[f] for f in FEATURES],
            "Typical phishing": [MEDIANS["phishing"][f] for f in FEATURES],
            "Typical legitimate": [MEDIANS["legitimate"][f] for f in FEATURES],
        })
        st.dataframe(table, hide_index=True, width="stretch")

    outside = [f for f in FEATURES
               if values[f] > RANGES[f]["max"] or values[f] < RANGES[f]["min"]]
    if outside:
        st.warning(
            "These values are outside the range seen during training, so the prediction is less "
            f"reliable: {outside}"
        )

    history = st.session_state.setdefault("history", [])
    history.insert(0, {
        "Input": source[:70], "Prediction": "PHISHING" if label_class == 0 else "LEGITIMATE",
        "Phishing prob. (%)": round(p_phish * 100, 1), "Risk": level,
    })
    del history[10:]


# ----------------------------------------------------------------------------
# Session defaults for the manual-entry mode
# ----------------------------------------------------------------------------
def preset_values(kind):
    return {f: int(round(MEDIANS[kind][f])) for f in FEATURES}


def apply_preset(kind):
    for f, v in preset_values(kind).items():
        st.session_state[f"in_{f}"] = v


for _f, _v in preset_values("legitimate").items():
    st.session_state.setdefault(f"in_{_f}", _v)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("About this model")
    st.write(f"**Algorithm:** {config['final_model']}")
    st.write(f"**Test accuracy:** {METRICS['accuracy'] * 100:.1f}%")
    st.write(f"**ROC-AUC:** {METRICS['roc_auc']:.3f}")
    st.write(f"**Trained on:** {config['data_summary']['train_rows']:,} URLs")
    st.caption("Trained in Google Colab. This app only loads the saved model.")
    if sklearn.__version__ != config.get("sklearn_version"):
        st.warning(
            f"scikit-learn here is {sklearn.__version__}, but the model was trained with "
            f"{config.get('sklearn_version')}. Install the same version to avoid problems."
        )
    st.divider()
    st.subheader("Recent checks")
    if st.session_state.get("history"):
        st.dataframe(pd.DataFrame(st.session_state["history"]), hide_index=True,
                     width="stretch")
    else:
        st.caption("Nothing checked yet.")

st.title("🔐 Phishing URL Detector")
st.caption("Machine-learning dashboard: Random Forest trained in Google Colab, served with Streamlit.")

tab_check, tab_perf, tab_imp, tab_about = st.tabs(
    ["🔎 Check URL", "📊 Model performance", "🌲 Feature importance", "ℹ️ About & limits"]
)

# ----------------------------------------------------------------------------
# Tab 1: check a URL / feature values
# ----------------------------------------------------------------------------
with tab_check:
    mode = st.radio(
        "Input mode",
        ["Paste a URL (approximate features)", "Enter the 13 feature values (exact)"],
        horizontal=True,
    )

    if mode.startswith("Paste"):
        st.info(
            "The model was trained on 13 pre-computed numbers, and the dataset does not document how "
            "two of them (`valid_url`, `nb_or`) were calculated. The other 11 are computed from your "
            "URL by simple counting. For these two, you choose the values below."
        )
        url_input = st.text_input("Enter a website URL", placeholder="https://example.com/login")
        with st.expander("Settings for the two undocumented features", expanded=False):
            valid_choice = st.selectbox(
                "valid_url", [0, 1], index=0,
                help="Most phishing rows in the training data have 0; 0 is also the most common value overall.")
            nb_or_choice = st.number_input("nb_or", min_value=0, max_value=50, value=0, step=1)

        if st.button("Check URL", type="primary"):
            clean_url, error = validate_url(url_input)
            if error:
                st.warning(error)
            else:
                values = extract_url_features(clean_url)
                values["valid_url"] = int(valid_choice)
                values["nb_or"] = int(nb_or_choice)
                if not clean_url.lower().startswith(("http://", "https://")):
                    st.caption("No http:// or https:// given, so isHttps was set to 0.")
                show_result(values, clean_url)
    else:
        st.write("Enter the values exactly as in the training dataset, or load an example.")
        b1, b2, _ = st.columns([1, 1, 3])
        b1.button("Load phishing-like example", on_click=apply_preset, args=("phishing",))
        b2.button("Load legitimate-like example", on_click=apply_preset, args=("legitimate",))

        values = {}
        cols = st.columns(4)
        for i, f in enumerate(FEATURES):
            with cols[i % 4]:
                if f in BINARY:
                    values[f] = st.selectbox(f, [0, 1], key=f"in_{f}", help=FEATURE_HELP.get(f))
                else:
                    values[f] = st.number_input(
                        f, min_value=0, max_value=100000, step=1, key=f"in_{f}",
                        help=FEATURE_HELP.get(f))

        if st.button("Check features", type="primary"):
            show_result({f: int(v) for f, v in values.items()}, "manual feature values")

# ----------------------------------------------------------------------------
# Tab 2: model performance (all numbers come from the Colab evaluation)
# ----------------------------------------------------------------------------
with tab_perf:
    st.subheader("Evaluation on the held-out test set")
    ds = config["data_summary"]
    st.caption(f"{ds['test_rows']} test URLs the model never saw during training "
               f"(dataset after cleaning: {ds['rows_after_cleaning']:,} rows).")

    m = st.columns(5)
    m[0].metric("Accuracy", f"{METRICS['accuracy'] * 100:.1f}%")
    m[1].metric("Precision (phishing)", f"{METRICS['precision_phishing'] * 100:.1f}%")
    m[2].metric("Recall (phishing)", f"{METRICS['recall_phishing'] * 100:.1f}%")
    m[3].metric("F1 (phishing)", f"{METRICS['f1_phishing'] * 100:.1f}%")
    m[4].metric("ROC-AUC", f"{METRICS['roc_auc']:.3f}")
    st.caption(f"Always guessing 'phishing' would score "
               f"{METRICS['majority_class_baseline_accuracy'] * 100:.1f}% accuracy (baseline).")

    c1, c2 = st.columns(2)
    with c1:
        cm = np.array(METRICS["confusion_matrix_phishing_legitimate"])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 16},
                    xticklabels=["Phishing", "Legitimate"], yticklabels=["Phishing", "Legitimate"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion matrix")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        fpr, tpr = config["roc_curve"]["fpr"], config["roc_curve"]["tpr"]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="tab:blue", linewidth=2.5, label=f"Random Forest (AUC = {METRICS['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random guessing")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve")
        ax.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    c3, c4 = st.columns(2)
    with c3:
        counts = ds["class_counts_after_cleaning"]
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.bar(["Phishing", "Legitimate"], [counts["phishing"], counts["legitimate"]],
               color=["#d62728", "#2ca02c"], edgecolor="black")
        for i, v in enumerate([counts["phishing"], counts["legitimate"]]):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom")
        ax.set_title("Class distribution (after cleaning)")
        ax.set_ylabel("URLs")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with c4:
        comp = pd.DataFrame(config["model_comparison_test"])
        melted = comp.melt(id_vars="Model", value_vars=["Accuracy", "F1 (phishing)", "ROC-AUC"],
                           var_name="Metric", value_name="Score")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        sns.barplot(data=melted, x="Metric", y="Score", hue="Model", palette="Set2", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_title("Model comparison (test set)")
        ax.legend(fontsize=7, loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Overfitting check")
    o = config["overfitting_check"]
    oc = st.columns(3)
    oc[0].metric("Training accuracy", f"{o['train_accuracy'] * 100:.1f}%")
    oc[1].metric("Cross-validation accuracy", f"{o['cv_accuracy'] * 100:.1f}%")
    oc[2].metric("Test accuracy", f"{o['test_accuracy'] * 100:.1f}%")
    st.warning(
        f"Training accuracy is {(o['train_accuracy'] - o['test_accuracy']) * 100:.1f} points above test "
        "accuracy, so the Random Forest partly memorised the training data. The test and "
        "cross-validation scores are the realistic ones."
    )

    with st.expander("Cross-validation and model comparison tables"):
        st.dataframe(pd.DataFrame(config["cross_validation"]), hide_index=True, width="stretch")
        st.dataframe(comp.round(4), hide_index=True, width="stretch")

# ----------------------------------------------------------------------------
# Tab 3: feature importance
# ----------------------------------------------------------------------------
with tab_imp:
    st.subheader("Which features does the model rely on?")
    c1, c2 = st.columns(2)
    with c1:
        d = imp_df.sort_values("importance")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(d["feature"], d["importance"], color="steelblue")
        ax.set_title("Random Forest importance (training)")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        d = perm_df.sort_values("accuracy_drop")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(d["feature"], d["accuracy_drop"], xerr=d["std"], color="darkorange",
                ecolor="black", capsize=3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Permutation importance (test set)")
        ax.set_xlabel("Accuracy drop when feature is shuffled")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    top = perm_df.sort_values("accuracy_drop", ascending=False).iloc[0]
    st.info(
        f"`{top['feature']}` matters most (shuffling it lowers test accuracy by about "
        f"{top['accuracy_drop'] * 100:.1f} points). Features with bars near zero add little."
    )

# ----------------------------------------------------------------------------
# Tab 4: about
# ----------------------------------------------------------------------------
with tab_about:
    st.subheader("How it works")
    st.markdown(
        """
1. Training, preprocessing and evaluation happened in **Google Colab**. This app never trains.
2. The saved Random Forest (`phishing_model.pkl`) and its feature order (`feature_names.pkl`) are loaded.
3. Your input becomes 13 numbers, in the training column order, and the model returns probabilities.
4. The label is **PHISHING URL** or **LEGITIMATE URL**; risk level comes from the phishing probability.
"""
    )
    st.subheader("Limits you should know")
    st.markdown(
        f"""
- **Moderate accuracy:** about {METRICS['accuracy'] * 100:.0f}% on the test set. About
  {(1 - METRICS['recall_legitimate']) * 100:.0f}% of legitimate test URLs were wrongly flagged, and
  {(1 - METRICS['recall_phishing']) * 100:.0f}% of phishing URLs were missed.
- **URL mode is approximate.** The dataset only contains pre-computed counts, and does not document how
  `valid_url` and `nb_or` were produced, so you set those two yourself. Other counts follow simple
  rules (for example `nb_dots` = number of '.' characters) and may differ slightly from the dataset's own.
- **Small dataset:** {ds['rows_after_cleaning']:,} unique rows after removing duplicates.
- Never rely on this tool alone to decide whether a website is safe.
"""
    )
    st.subheader("Feature meanings used in this app")
    st.dataframe(pd.DataFrame({"Feature": list(FEATURE_HELP), "Meaning": list(FEATURE_HELP.values())}),
                 hide_index=True, width="stretch")