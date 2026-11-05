"""Trains and compares Naive Bayes, Logistic Regression and Random Forest on AG News.

    python train_model.py            # full dataset (120,000 articles)
    python train_model.py --quick    # 30,000-article subset (much faster)
"""
import argparse
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.naive_bayes import MultinomialNB

from utils import CATEGORY_NAMES, clean_text

DATA_DIR, MODEL_DIR, OUTPUT_DIR = "data", "models", "outputs"
RANDOM_STATE = 42


def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' not found. Run 'python download_dataset.py' first.")
    df = pd.read_csv(path, header=None, names=["label", "title", "description"])
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    df["category"] = df["label"].map(CATEGORY_NAMES)
    return df


def evaluate(name, model, X_test, y_test, labels):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0)
    safe = name.lower().replace(" ", "_")

    report = classification_report(y_test, y_pred, target_names=labels, digits=4)
    with open(os.path.join(OUTPUT_DIR, f"classification_report_{safe}.txt"),
              "w", encoding="utf-8") as f:
        f.write(f"{name}\n{'=' * len(name)}\n\n{report}")

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted category")
    plt.ylabel("Actual category")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{safe}.png"), dpi=150)
    plt.close()

    print(f"\n--- {name} ---\n{report}")
    return {"Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1-Score": f1}


def main(quick):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset ...")
    train_df = load_dataset(os.path.join(DATA_DIR, "train.csv"))
    test_df = load_dataset(os.path.join(DATA_DIR, "test.csv"))
    if quick:
        train_df = train_df.sample(30000, random_state=RANDOM_STATE)
        print("QUICK MODE: using a 30,000-article training subset.")
    print(f"Training articles: {len(train_df)} | Test articles: {len(test_df)}")
    print("\nClass distribution (training):")
    print(train_df["category"].value_counts().to_string())

    print("\nCleaning text ...")
    train_df["clean"] = train_df["text"].apply(clean_text)
    test_df["clean"] = test_df["text"].apply(clean_text)

    print("Extracting TF-IDF features ...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000,
                                 ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_df["clean"])   # fit on TRAIN only
    X_test = vectorizer.transform(test_df["clean"])
    y_train, y_test = train_df["category"], test_df["category"]
    print(f"Feature matrix shape: {X_train.shape}")

    labels = ["World", "Sports", "Business", "Sci/Tech"]
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=5,
                                                  random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=3,
                                                n_jobs=-1, random_state=RANDOM_STATE),
    }

    results, trained = [], {}
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        start = time.time()
        model.fit(X_train, y_train)
        seconds = time.time() - start
        print(f"Training time: {seconds:.1f} seconds")
        row = evaluate(name, model, X_test, y_test, labels)
        row["Training Time (s)"] = round(seconds, 1)
        results.append(row)
        trained[name] = model
        joblib.dump(model, os.path.join(
            MODEL_DIR, name.lower().replace(" ", "_") + ".joblib"), compress=3)

    results_df = pd.DataFrame(results).round(4)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    print("\n================ MODEL COMPARISON ================")
    print(results_df.to_string(index=False))

    plt.figure(figsize=(7, 4))
    bars = plt.bar(results_df["Model"], results_df["Accuracy"] * 100,
                   color=["#4C78A8", "#54A24B", "#F58518"])
    plt.bar_label(bars, fmt="%.2f%%")
    plt.ylim(80, 100)
    plt.ylabel("Test accuracy (%)")
    plt.title("Accuracy Comparison on AG News Test Set")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_comparison.png"), dpi=150)
    plt.close()

    best_name = results_df.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
    joblib.dump(trained[best_name], os.path.join(MODEL_DIR, "best_model.joblib"), compress=3)
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump({"best_model": best_name, "train_size": len(train_df),
                   "test_size": len(test_df), "quick_mode": quick,
                   "num_features": int(X_train.shape[1])}, f, indent=2)

    print(f"\nBest model: {best_name}")
    print("Saved models to 'models/' and results to 'outputs/'.")
    print("Next step: streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AG News classifiers")
    parser.add_argument("--quick", action="store_true",
                        help="train on a 30,000-article subset (faster)")
    main(parser.parse_args().quick)