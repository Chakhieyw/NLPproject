# =====================================================
# Document Classification using NLP (Robust Version)
# =====================================================

import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==============================
# 1) Load Dataset
# ==============================
df = pd.read_csv("Dataset-NlpProject - Sheet1.csv")

print("Dataset shape:", df.shape)
print(df.head())


# ==============================
# 2) Text Preprocessing
# ==============================

def clean_text(text):
    text = re.sub(r"[^\u0E00-\u0E7Fa-zA-Z\s]", "", str(text))
    return text

def remove_stopwords(text):
    stopwords = set(thai_stopwords())
    tokens = word_tokenize(text)
    return " ".join([w for w in tokens if w not in stopwords])

df['Text'] = df['Text'].apply(clean_text)
df['Text_clean'] = df['Text'].apply(remove_stopwords)

print("\nตัวอย่างหลังทำความสะอาด:")
print(df[['Text', 'Text_clean']].head())


# ==============================
# 3) Smart Train-Test Split
# ==============================

X = df['Text_clean']
y = df['Label']

num_classes = y.nunique()
num_samples = len(df)

print("\nNumber of samples:", num_samples)
print("Number of classes:", num_classes)

# คำนวณ test_size แบบปลอดภัย
min_test_size = num_classes / num_samples

if min_test_size < 0.2:
    test_size = 0.2
else:
    test_size = min_test_size

print("Using test_size:", round(test_size, 2))

# ถ้าข้อมูลพอ → ใช้ stratify
if num_samples >= num_classes * 2:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )
else:
    # ถ้าน้อยเกินไป → ไม่ stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42
    )


# ==============================
# 4) TF-IDF
# ==============================
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ==============================
# 5) Train Models
# ==============================

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)


# ==============================
# 6) Evaluation
# ==============================

print("\n============================")
print("Naive Bayes Accuracy:",
      accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb, zero_division=0))

print("\n============================")
print("Logistic Regression Accuracy:",
      accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr, zero_division=0))


# ==============================
# 7) Confusion Matrix (NB)
# ==============================
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred_nb, labels=labels)

plt.figure()
sns.heatmap(cm,
            annot=True,
            fmt='d',
            xticklabels=labels,
            yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Naive Bayes)")
plt.show()


# ==============================
# 8) Predict New Text
# ==============================

def predict_text(text, model):
    text = clean_text(text)
    text = remove_stopwords(text)
    text_tfidf = vectorizer.transform([text])
    return model.predict(text_tfidf)[0]



