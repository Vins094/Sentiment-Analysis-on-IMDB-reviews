# Sentiment Analysis of IMDb Movie Reviews

## Overview

This project focuses on building a sentiment analysis model to predict the sentiment (positive or negative) of IMDb movie reviews. The goal is to assist viewers in making informed movie choices and provide filmmakers with insights into audience reactions.

## Problem Statement

Manually analyzing large volumes of textual reviews is challenging. This project aims to leverage NLP and machine learning to classify IMDb reviews as positive or negative, helping users make informed decisions and benefiting the entertainment industry.

## Research Hypothesis

We hypothesize that BERT outperforms traditional machine learning models that use feature extraction techniques such as n-grams, TF-IDF, Word2Vec, and GloVe. Due to its ability to comprehend context, BERT is expected to provide more accurate predictions.

## Approach & Methodology

### Exploratory Data Analysis (EDA)

  - Understanding patterns in positive and negative sentiment reviews.
  - Preprocessing text (removing stopwords, HTML tags, and handling punctuation and emojis).

### Baseline Models

  - Used Bag of Words (BoW), TF-IDF, and word embeddings.
  - Machine learning models: Logistic Regression, Naïve Bayes, and SVM.

### Deep Learning Approaches

   - Word Embedding techniques (GloVe, FastText).
   - LSTM and Bidirectional LSTM models.

### Transformer-Based Model (BERT)

  - Fine-tuning BERT for sentiment classification.
  - Evaluating model performance against traditional methods.

## Dataset

The dataset was sourced from Hugging Face and consists of:

  - 25k Training reviews, 25k Test reviews, 50k Unlabeled reviews
  - Each review is labeled (1: Positive, 0: Negative)

## Implementation

  - Preprocessing: Tokenization, lemmatization, and stopword removal.
  - Feature Engineering: BoW, TF-IDF, and word embeddings.

## Model Training & Evaluation:
 
  - Logistic Regression with TF-IDF achieved 90% accuracy.
  - RNN-Bidirectional LSTM with word embeddings performed poorly.
  - BERT fine-tuning achieved 94% accuracy.

## Results & Error Analysis

  - Traditional machine learning models (Logistic Regression + TF-IDF) performed well but lacked contextual understanding.
  - LSTM and Bidirectional LSTM failed to capture semantic meaning.
  - BERT outperformed all models but struggled with sarcasm detection.

## Future Work

Enhance the interpretability of deep learning models. Combining supervised and unsupervised learning to improve sarcasm detection. Incorporating sentiment lexicons and domain-specific embeddings.
