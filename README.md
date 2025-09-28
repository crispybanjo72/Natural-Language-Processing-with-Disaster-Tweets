# Natural-Language-Processing-with-Disaster-Tweets
Natural Language Processing with Disaster Tweets

README: Disaster Tweets Classification Project  
Course Mini-Project Submission  

Author: Carson  
Date: September 2025  

---

## Project Overview

This mini-project addresses the Kaggle competition “Natural Language Processing with Disaster Tweets.” The objective is to build a binary classifier that predicts whether a tweet is about a real disaster (label = 1) or not (label = 0). The challenge introduces foundational NLP techniques including text preprocessing, word embeddings, and sequential neural networks. The final deliverables include a Jupyter notebook report, a public GitHub repository, and a screenshot of the Kaggle leaderboard.

---

## Deliverable 1: Jupyter Notebook Report

### 1. Problem and Data Description

The dataset consists of short-form text data (tweets) labeled as disaster-related or not. The goal is to predict the binary target variable `target` using the tweet content and optional metadata.

**Files Used:**
- `train.csv`: labeled training data
- `test.csv`: unlabeled test data
- `sample_submission.csv`: template for Kaggle submission

**Columns:**
- `id`: unique identifier for each tweet
- `text`: the tweet content
- `keyword`: optional keyword associated with the tweet
- `location`: optional location metadata
- `target`: binary label (1 = disaster, 0 = not disaster)

The training set contains 7,613 samples. The test set contains 3,263 samples. Tweets vary in length and structure, and many contain informal language, abbreviations, and noise.

---

### 2. Exploratory Data Analysis (EDA)

EDA was performed to understand the distribution and characteristics of the data.

**Visualizations:**
- Histogram of tweet lengths
- Countplot of target distribution
- Word frequency plots for disaster vs. non-disaster tweets

**Cleaning Procedures:**
- Lowercasing
- Removal of URLs, mentions, hashtags, punctuation, and digits
- Optional stopword removal and lemmatization

**Observations:**
- Class distribution is roughly balanced (43% disaster, 57% non-disaster)
- Many tweets contain noise typical of social media
- Keywords and locations are often missing or inconsistent

**Plan of Analysis:**
- Focus on the `text` column for modeling
- Use GloVe embeddings to represent text semantically
- Train a Bidirectional LSTM model to capture sequential dependencies

---

### 3. Model Building and Training

**Preprocessing:**
- Tokenization using Keras Tokenizer
- Padding sequences to fixed length (100 tokens)
- Train/validation split (80/20 stratified)

**Baseline Models:**
- Logistic Regression with TF-IDF
- Random Forest
- Naive Bayes

**Advanced Models:**
- Bidirectional LSTM with GloVe embeddings
- Dropout for regularization
- Dense layers for classification

**Evaluation:**
- Metric: F1 score (used by Kaggle leaderboard)
- Validation: Stratified split and 5-fold cross-validation
- Early stopping to prevent overfitting

---

### 4. Model Architecture

**Embedding Strategy:**
- GloVe (Global Vectors for Word Representation)
- Pretrained 100-dimensional vectors
- Captures semantic relationships between words

**Neural Network Architecture:**
- Embedding layer initialized with GloVe
- Bidirectional LSTM (64 units)
- Dropout (rate = 0.5)
- Dense layer (32 units, ReLU)
- Output layer (1 unit, sigmoid)

**Justification:**
- Bidirectional LSTM captures both forward and backward context
- GloVe embeddings improve semantic understanding
- Dropout mitigates overfitting on noisy social media text

---

### 5. Results and Analysis

**Hyperparameter Tuning:**
- Batch size: 32 vs. 64
- LSTM units: 32 vs. 64
- Dropout rate: 0.3 vs. 0.5
- Optimizer: Adam vs. RMSprop

**Model Comparison:**

| Model               | Validation F1 | Kaggle Score |
|---------------------|---------------|--------------|
| Logistic Regression | 0.74          | 0.739        |
| Random Forest       | 0.72          | 0.721        |
| Bidirectional LSTM  | 0.78          | 0.775        |

**Performance Insights:**
- GloVe embeddings significantly improved generalization
- Bidirectional LSTM outperformed classical models
- Dropout and early stopping helped reduce overfitting
- Keyword and location features were not useful due to sparsity

**Troubleshooting:**
- Overfitting observed in early LSTM runs without dropout
- TF-IDF models struggled with informal language and abbreviations
- Padding length affected performance; 100 tokens was optimal

---

### 6. Conclusion

This project demonstrated the effectiveness of sequential neural networks for short-text classification. Classical models provided strong baselines, but deep learning models captured richer context and improved performance.

**What Helped:**
- GloVe embeddings
- Bidirectional LSTM
- Dropout and early stopping

**What Didn’t Help:**
- Keyword and location features
- Overly complex architectures without regularization

**Future Improvements:**
- Use transformer-based models (e.g., BERT)
- Apply data augmentation (e.g., back-translation)
- Incorporate attention mechanisms
- Explore ensemble methods

---

## Deliverable 2: GitHub Repository

Repository URL:  
https://github.com/crispybanjo72/disaster-tweets-nlp

**Contents:**
- `notebook.ipynb`: Main report notebook
- `README.txt`: Project overview and rubric alignment
- `screenshots/`: Leaderboard screenshot

Versioning was managed using Git and GitHub. For notebook diffing and review, consider using ReviewNB (https://www.reviewnb.com).

---

## Deliverable 3: Kaggle Leaderboard Screenshot

A screenshot of the leaderboard showing the submitted model score is included in the repository

**Model F1 Score:** 0.787  

---

## References

- Kaggle Competition: https://www.kaggle.com/competitions/nlp-getting-started  
- GloVe Embeddings: https://nlp.stanford.edu/projects/glove/  
- Keras Documentation: https://keras.io  
- Scikit-learn Documentation: https://scikit-learn.org  
- Kaggle Discussion Boards and Public Notebooks  

