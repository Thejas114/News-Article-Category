📌 Project Overview

News Article Category Prediction is an NLP-based machine learning application that automatically classifies news articles into predefined categories based on their textual content.

The system uses Natural Language Processing techniques and machine learning algorithms to analyze news headlines or full articles and predict their appropriate categories.

🎯 Objectives

Automate news article classification using NLP.

Preprocess textual data and extract meaningful features using TF-IDF.

Train and compare multiple machine learning classification algorithms.

Evaluate model performance using standard evaluation metrics.

Provide an interactive web interface for news category prediction.

📂 News Categories

The system classifies news articles into four categories using the AG News dataset:

World – International news and global events.

Sports – Sports events, matches, and tournaments.

Business – Business, finance, and economic news.

Sci/Tech – Science, technology, and innovations.

🛠️ Technologies Used

Programming Language: Python

NLP: Text Preprocessing, Tokenization, TF-IDF

Machine Learning: Scikit-learn

Algorithms: Multinomial Naive Bayes, Logistic Regression, Random Forest

Data Processing: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Web Framework: Streamlit

Model Storage: Joblib

⚙️ Project Workflow

Dataset Collection: Load the AG News dataset containing labeled news articles.

Text Preprocessing: Clean text, remove unnecessary characters, and process textual data.

Feature Extraction: Apply TF-IDF vectorization to convert text into numerical features.

Model Training: Train Naive Bayes, Logistic Regression, and Random Forest classifiers.

Model Evaluation: Compare models using Accuracy, Precision, Recall, and F1-score.

Prediction: Predict the category of newly entered news articles.

Web Application: Display predictions through an interactive Streamlit interface.

📊 Machine Learning Algorithms

Multinomial Naive Bayes

Logistic Regression

Random Forest Classifier

The models are compared using standard classification metrics to identify the best-performing model.

🚀 Features

News headline and article classification.

Automated text preprocessing.

TF-IDF feature extraction.

Multiple machine learning model comparison.

Performance evaluation and visualization.

Interactive Streamlit web interface.

Real-time category prediction.

📁 Project Structure

news_article_category_prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── data/
│   └── dataset.csv
│
├── models/
│   ├── best_model.joblib
│   └── tfidf_vectorizer.joblib
│
├── outputs/
│   ├── category_distribution.png
│   ├── model_accuracy_comparison.png
│   └── confusion_matrix.png
│
└── reports/
    └── classification_report.txt

💻 Installation and Setup

1. Clone the repository

git clone https://github.com/Thejas114/News-Article-Category.git

2. Navigate to the project directory

cd news_article_category_prediction

3. Create a virtual environment

python -m venv venv

4. Activate the virtual environment (Windows)

venv\Scripts\activate

5. Install dependencies

pip install -r requirements.txt

6. Prepare the dataset

Download the AG News dataset from:

https://huggingface.co/datasets/fancyzhx/ag_news

Follow the dataset preparation instructions and place the dataset in the appropriate directory.


7. Train the machine learning models

python train_model.py

8. Run the Streamlit application

streamlit run app.py

The application will open in your default web browser.

📌 Example Prediction

Input:

"India wins the cricket championship after defeating Australia in the final."

Output:

Predicted Category: Sports

📈 Model Evaluation

The performance of the trained models is evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Actual evaluation results are generated during model training and testing.

🔮 Future Enhancements

Support additional news categories.

Integrate advanced NLP models such as BERT.

Support multilingual news classification.

Integrate live news feeds for automatic categorization.

Develop personalized news recommendation features.

👨‍💻 Project Purpose

This project was developed as a Computer Science and Engineering mini project to demonstrate the practical application of Natural Language Processing and Machine Learning in automated text classification.
