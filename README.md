# 📰 Semantic Fake News Detection

A machine learning project that classifies news articles as *real* or *fake* using Word2Vec embeddings and semantic features such as POS tagging and lemmatization. This project applies Logistic Regression, Decision Tree, and Random Forest classifiers and evaluates them using key metrics like Accuracy and F1 Score.

---

## 📌 Problem Statement

With the rise of misinformation, it's critical to develop tools that can automatically distinguish between fake and real news. This project aims to leverage semantic understanding through natural language processing (NLP) to improve fake news detection accuracy.

---

## 🧠 Methodology

1. **Data Preprocessing**
   - Lowercasing
   - Removing stopwords
   - Lemmatization
   - POS tagging (keeping only nouns)

2. **Feature Engineering**
   - Word2Vec embeddings using the pre-trained `word2vec-google-news-300` model.
   - Averaged vector representation for each news article.

3. **Model Building**
   - Models used:
     - Logistic Regression
     - Decision Tree
     - Random Forest
   - Evaluation Metrics:
     - Accuracy
     - Precision
     - Recall
     - F1 Score

4. **Hyperparameter Tuning**
   - Used `RandomizedSearchCV` for tuning Random Forest parameters.

---

## 📊 Results

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 0.9095   | 0.9008    | 0.9105 | 0.9056   |
| Random Forest      | 0.9094   | 0.9107    | 0.8981 | 0.9043   |
| Decision Tree      | 0.8213   | 0.8308    | 0.7851 | 0.8073   |

**Best Model**: Logistic Regression based on highest F1 Score  
**Priority Metric**: F1 Score — balances both precision and recall

---

## 📌 Key Insights

- Fake news tends to overuse general nouns and certain repeated claims.
- Semantic features (lemmatized nouns only) improved signal-to-noise ratio.
- Logistic Regression offered the most consistent and interpretable results.

---

## 📦 Tech Stack

- Python 🐍
- Scikit-learn
- Gensim (for Word2Vec)
- Pandas & NumPy
- NLTK / spaCy
- Matplotlib / Seaborn (for visualizations)

---

## 🚀 How to Run

```bash
# Clone this repository
git clone https://github.com/yourusername/semantic-fake-news-detector.git
cd semantic-fake-news-detector

# Install dependencies
pip install -r requirements.txt

# Run the notebook or script
```

---

## 📂 Project Structure

```
📁 data/                 # Raw and cleaned datasets
📁 models/               # Saved model files (optional)
📁 notebooks/            # Jupyter notebooks for EDA and modeling
📄 main.py               # (optional) script to train models
📄 README.md             # This file
```
