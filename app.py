"""Streamlit app: streamlit run app.py"""
import json
import os

import joblib
import pandas as pd
import streamlit as st

from utils import CATEGORY_ICONS, clean_text

MODEL_DIR, OUTPUT_DIR = "models", "outputs"
MODEL_FILES = {
    "Multinomial Naive Bayes": "multinomial_naive_bayes.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
}
SAMPLES = {
    "-- choose an example --": "",
    "Sports": "Messi scores twice as Argentina beat Brazil in World Cup qualifier thriller",
    "Business": "Stocks rally as Federal Reserve signals interest rate cuts and oil prices fall",
    "Sci/Tech": "Researchers unveil new artificial intelligence chip that doubles smartphone battery life",
    "World": "United Nations leaders meet to discuss ceasefire and humanitarian aid for the region",
}

st.set_page_config(page_title="News Category Predictor", page_icon="📰", layout="wide")


@st.cache_resource
def load_vectorizer():
    return joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))


@st.cache_resource
def load_model(file_name):
    return joblib.load(os.path.join(MODEL_DIR, file_name))


def files_ready():
    needed = ["tfidf_vectorizer.joblib"] + list(MODEL_FILES.values())
    return all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in needed)


st.title("📰 News Article Category Prediction")
st.caption("NLP + Machine Learning | TF-IDF | AG News | World · Sports · Business · Sci/Tech")

if not files_ready():
    st.error("Trained models were not found in the 'models' folder.")
    st.info("Run first:\n\n`python download_dataset.py`\n\n`python train_model.py`")
    st.stop()

vectorizer = load_vectorizer()

st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.selectbox("Choose ML model", list(MODEL_FILES.keys()), index=1)
model = load_model(MODEL_FILES[model_name])
st.sidebar.markdown("---")
st.sidebar.markdown("**How it works**\n\n1. Text is cleaned\n2. TF-IDF turns words into numbers\n"
                    "3. The chosen model predicts the category")

tab_predict, tab_perf = st.tabs(["🔮 Predict", "📊 Model Performance"])

with tab_predict:
    example = st.selectbox("Try an example:", list(SAMPLES.keys()))
    user_text = st.text_area("Enter a news headline or article:", value=SAMPLES[example],
                             height=180, placeholder="Paste a news headline or article here...")
    if st.button("Predict Category", type="primary"):
        cleaned = clean_text(user_text)
        if len(cleaned.split()) < 3:
            st.warning("Please enter at least 3 words.")
        else:
            features = vectorizer.transform([cleaned])
            prediction = model.predict(features)[0]
            probs = model.predict_proba(features)[0]
            confidence = probs.max() * 100
            st.success(f"### {CATEGORY_ICONS.get(prediction, '')} Predicted category: **{prediction}**")
            st.metric("Confidence", f"{confidence:.1f}%")
            if confidence < 50:
                st.info("The model is not very sure. The text may mix several topics.")
            prob_df = (pd.DataFrame({"Category": model.classes_,
                                     "Probability (%)": (probs * 100).round(2)})
                       .sort_values("Probability (%)", ascending=False))
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Probability for each category**")
                st.dataframe(prob_df, hide_index=True, use_container_width=True)
            with c2:
                st.bar_chart(prob_df.set_index("Category"))
            with st.expander("See the cleaned text used by the model"):
                st.write(cleaned)

with tab_perf:
    csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    if not os.path.exists(csv_path):
        st.warning("Run `python train_model.py` to generate evaluation results.")
    else:
        meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            st.write(f"Trained on **{meta['train_size']:,}** articles, tested on "
                     f"**{meta['test_size']:,}** unseen articles. Best model: **{meta['best_model']}**.")
        st.subheader("Model comparison (test set)")
        results = pd.read_csv(csv_path)
        st.dataframe(results, hide_index=True, use_container_width=True)
        acc_img = os.path.join(OUTPUT_DIR, "accuracy_comparison.png")
        if os.path.exists(acc_img):
            st.image(acc_img, caption="Accuracy comparison")
        st.subheader("Confusion matrices")
        cols = st.columns(3)
        for col, name in zip(cols, MODEL_FILES):
            img = os.path.join(OUTPUT_DIR, "confusion_matrix_" + name.lower().replace(" ", "_") + ".png")
            if os.path.exists(img):
                col.image(img, caption=name)
        st.subheader("Per-class report")
        choice = st.selectbox("Select model", list(MODEL_FILES.keys()), key="rep")
        rep = os.path.join(OUTPUT_DIR, "classification_report_" + choice.lower().replace(" ", "_") + ".txt")
        if os.path.exists(rep):
            with open(rep, encoding="utf-8") as f:
                st.code(f.read())