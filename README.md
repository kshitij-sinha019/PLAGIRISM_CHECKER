# PLAGIRISM_CHECKER
https://colab.research.google.com/drive/1Gw3lnpcAy2Wh5j4K4ivpuVuyh9mPmzqP?usp=sharing


# Plagiarism Detection Using Statistical Language Model

This project implements a plagiarism detection system using a Statistical Language Model (SLM) with Scikit-Learn and TF-IDF vectorization. The model classifies whether two sentences in a pair are plagiarized (i.e., the same or different).

## Project Overview

The plagiarism detection model uses a dataset from Kaggle's MIT Plagiarism Detection dataset and applies the following steps:

1. **Data Preprocessing**: The dataset is cleaned and preprocessed by:
   - Lowercasing text
   - Removing stopwords
   - Tokenizing and stemming text
   - Combining sentence pairs for classification
   
2. **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) is used for vectorizing text, along with n-grams (1-3 grams).

3. **Model Training**: An SGDClassifier (Stochastic Gradient Descent) is used for classification, and hyperparameter tuning is done using GridSearchCV.

4. **Evaluation**: The model is evaluated using accuracy scores and a classification report, and it is saved for future use.

## Requirements

- Python 3
- Libraries:
  - transformers
  - datasets
  - scikit-learn
  - pandas
  - numpy
  - torch
  - nltk
  - joblib

## Installation

To install the required libraries, run the following command:

```bash
pip install transformers datasets scikit-learn pandas numpy torch nltk joblib
