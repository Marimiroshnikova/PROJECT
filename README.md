# Binary Sentiment Classification Project

## Table of Contents
- [Project Overview](#project-overview)
- [Data Science Part](#data-science-part)
  - [Introduction](#introduction)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Text Preprocessing & Feature Engineering](#text-preprocessing--feature-engineering)
  - [Modeling & Evaluation](#modeling--evaluation)
  - [Potential Business Applications](#potential-business-applications)
- [Machine Learning Engineering (MLE) Part](#machine-learning-engineering-mle-part)
  - [Repository Structure](#repository-structure)
  - [How to Run the Project via Docker](#how-to-run-the-project-via-docker)
- [Dependencies](#dependencies)

---

## Project Overview

This project demonstrates a complete end-to-end solution for binary sentiment classification using movie reviews. It is divided into two parts:
- **DS Part:** Data exploration, text preprocessing (tokenization, stop-words filtering, stemming vs. lemmatization, vectorization), model building, and evaluation.
- **MLE Part:** Packaging the solution into a reproducible repository with Dockerized training and inference pipelines.

---

## Data Science Part

### Introduction

The goal of this project is to classify movie reviews as either **positive** or **negative**. We use a dataset consisting of 50,000 reviews (40,000 for training and 10,000 for testing) and aim for a minimum accuracy of 85%.

### Exploratory Data Analysis (EDA)

- **Data Overview:**  
  - **Training Data Shape:** Initially 40,000 reviews and 2 columns (review and sentiment).  
  - **Duplicates:** 272 duplicate rows were found and removed.
  - **Missing Values:** There are no missing values.
  - **Sentiment Distribution:** The distribution is balanced with approximately 19,900 positive and 19,800 negative reviews.

- **Review Length Analysis:**  
  - A new column `review_length` was created to capture the number of words per review.
  - Summary statistics showed an average review length of ~231 words, with a range from 4 to 2470 words.
  - A histogram was plotted to visualize the distribution of review lengths.

- **Word Frequency & N-Grams:**  
  - The most common words (e.g., "movie", "film", "good") were identified using a word frequency count.
  - Bigrams and trigrams were extracted and visualized to understand common phrase patterns in the reviews.
  - Word clouds were generated separately for positive and negative reviews.

### Text Preprocessing & Feature Engineering

- **Cleaning & Tokenization:**  
  - HTML tags were removed using BeautifulSoup.
  - Reviews were converted to lowercase, and extra spaces were removed.
  - Tokenization was done using NLTK’s `word_tokenize`, retaining only alphabetic tokens.
  - Stop-words were filtered out using NLTK's English stopwords list.

- **Stemming vs. Lemmatization:**  
  - Both stemming (using `PorterStemmer`) and lemmatization (using `WordNetLemmatizer`) were applied.
  - A comparison on sample tokens was performed to understand the differences:
    - **Original Tokens:** `['caught', 'little', 'gem', ...]`
    - **Stemmed Tokens:** `['caught', 'littl', 'gem', ...]`
    - **Lemmatized Tokens:** `['caught', 'little', 'gem', ...]`
  - Based on the analysis, lemmatization was used for producing the final tokenized reviews.

- **Vectorization:**  
  - Two approaches were compared:
    - **CountVectorizer:** Considered unigrams and bigrams.
    - **TfidfVectorizer:** Also used n-gram range (1, 2) to transform the text.
  - The TF-IDF representation was chosen as the primary feature set for modeling.

### Modeling & Evaluation

- **Models Evaluated:**  
  Three baseline models were built and evaluated:
  - **Logistic Regression:** Accuracy ≈ 88.41%
  - **Multinomial Naive Bayes:** Accuracy ≈ 89.00%
  - **Linear SVM:** Accuracy ≈ 90.30%

- **Model Selection:**  
  - **Linear SVM** was selected as the best model based on the highest accuracy (90.30%), which exceeds the minimum required threshold of 85%.
  - The best model and the TF-IDF vectorizer were saved as pickle files for later inference.

### Potential Business Applications

- **Customer Feedback Analysis:** Automatically analyzing movie reviews to gauge customer sentiment.
- **Social Media Monitoring:** Tracking public sentiment towards films or related content on social media.
- **Recommendation Systems:** Integrating sentiment analysis into recommendation engines to better align with user preferences.
- **Marketing & PR:** Identifying negative feedback trends to proactively manage brand reputation.

---

## Machine Learning Engineering (MLE) Part

### Repository Structure

The repository is organized as follows:

```
/PROJECT/
├── data/            
│   ├── raw/
│        ├── test.csv
│        ├── train.csv
│   └── processed/
├── notebooks/       
│   ├── DS.ipynb
├── src/				
│   ├── train/
│   │   ├── train.py
│   │   └── Dockerfile
│   ├── inference/
│   │   ├── run_inference.py
│   │   └── Dockerfile
│   └── data_loader.py
├── outputs/
│   ├── models/
│        ├── best_model.pkl
│        ├── tfidf_vectorizer.pkl
│   ├── predictions/
│        ├── predictions.csv
│   └── figures/
├── README.md
├── requirements.txt
├── .conda
├── env
└── .gitignore 		

```

