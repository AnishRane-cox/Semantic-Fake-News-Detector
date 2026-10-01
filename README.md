# 📰 Semantic Fake News Detection with Word2Vec

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-spaCy%20%7C%20NLTK-09A3D5?style=flat-square)
![gensim](https://img.shields.io/badge/gensim-Word2Vec%20(Google%20News%20300d)-4B8BBE?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Classification-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/AnishRane-cox/Semantic-Fake-News-Detector/blob/main/Fake_News_Detection.ipynb)

> Classifying **~45,000 news articles as true or fake** based on their **meaning**, not just keywords — using noun-focused lemmatisation and pre-trained **Word2Vec** embeddings.

| Model | Accuracy | Precision | Recall | **F1** |
|---|---|---|---|---|
| **Logistic Regression** | 0.904 | 0.894 | **0.906** | **0.900** |
| Random Forest | **0.907** | **0.910** | 0.893 | **0.902** |
| Decision Tree | 0.824 | 0.830 | 0.793 | 0.811 |

*Evaluated on a stratified 30% validation split.*

---

## 📌 Problem

Misinformation spreads faster than it can be fact-checked manually. A system that flags likely-fake articles automatically helps media platforms protect credibility and prioritise human review.

Keyword models (Bag-of-Words, TF-IDF) treat words as unrelated tokens. **Semantic** models use embeddings in which similar words sit close together, so the classifier can generalise from *"scandal"* to *"cover-up"* even if one of them was never seen during training.

## 🗂️ Data

| File | Articles | Columns |
|---|---|---|
| `True.csv` | 21,417 | title, text, date |
| `Fake.csv` | 23,502 | title, text, date |

## 🔬 Pipeline

```mermaid
flowchart LR
    A[True + Fake CSVs] --> B[Label & merge<br/>title + text]
    B --> C[Clean text<br/>lowercase, strip brackets,<br/>punctuation, digits]
    C --> D[POS-tag + lemmatise<br/>keep nouns NN/NNS,<br/>drop stop-words]
    D --> E[70/30 stratified split]
    E --> F[EDA: lengths, top words,<br/>uni/bi/tri-grams, word clouds]
    F --> G[Word2Vec Google-News-300<br/>average article vector]
    G --> H[LogReg · Decision Tree · Random Forest]
```

1. **Preparation** – labelled both datasets, merged them and combined `title` + `text` into one field.
2. **Cleaning** – lowercase, remove bracketed text, punctuation and words containing numbers.
3. **Linguistic filtering** – spaCy POS tagging + lemmatisation, keeping only **nouns** (`NN`, `NNS`) — they carry the topic and entities of an article.
4. **EDA** – character-length distributions before/after processing, top-40 words and top uni-/bi-/tri-grams for each class.
5. **Features** – each article becomes the **average of its 300-d Word2Vec vectors** (pre-trained `word2vec-google-news-300`).
6. **Models** – Logistic Regression, Decision Tree and Random Forest.

## 📈 Results

- **Logistic Regression and Random Forest are practically tied (F1 ≈ 0.90).** Logistic Regression was chosen as the final model: it is simpler, faster and has slightly higher recall.
- **F1 was prioritised** because both errors are costly — flagging real news damages trust, while missing fake news lets misinformation spread.
- A single Decision Tree trails by ~9 F1 points — ensembles and linear models handle dense embeddings better.

**Patterns found in EDA:** true news leans on institutional, fact-based vocabulary (officials, statements, reports), while fake news uses more emotionally charged, sensational terms and vaguer sourcing.

## 🧠 What I Learned

- Building an NLP pipeline end-to-end: cleaning → POS filtering → embeddings → classification.
- Using **pre-trained embeddings** to inject semantic knowledge into a small model.
- Why simple averaged embeddings + a linear model make a strong, explainable baseline.

## 🔭 Next Steps

- Compare with a TF-IDF baseline and a fine-tuned transformer (e.g. DistilBERT).
- Keep verbs/adjectives too — sentiment and style words may carry extra signal.
- Test on articles from other publishers and periods to measure robustness to topic shift.

## 🚀 How to Run

Click **Open in Colab** above, or run locally:

```bash
git clone https://github.com/AnishRane-cox/Semantic-Fake-News-Detector.git
cd Semantic-Fake-News-Detector
pip install pandas numpy nltk spacy gensim scikit-learn matplotlib seaborn wordcloud jupyter
python -m spacy download en_core_web_sm
jupyter notebook Fake_News_Detection.ipynb
```

> The first run downloads the ~1.6 GB Google-News Word2Vec model via `gensim.downloader`.

## 📁 Repository Structure

```
├── Fake_News_Detection.ipynb                               # Full pipeline
├── True.csv / Fake.csv                                     # Data
├── Fake News Classification Using Semantic Text Analysis.pdf  # Report
└── README.md
```

---

## 👤 Author

**Anish Rane** — Data & AI Engineer · MSc Machine Learning & AI (LJMU) · Mechanical Engineer

[![Portfolio](https://img.shields.io/badge/Portfolio-1D9E75?style=flat-square&logo=githubpages&logoColor=white)](https://anishrane-cox.github.io/Portfolio/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anish-rane/)
[![GitHub](https://img.shields.io/badge/GitHub-AnishRane--cox-181717?style=flat-square&logo=github)](https://github.com/AnishRane-cox)

⭐ If you found this useful, consider starring the repo.
