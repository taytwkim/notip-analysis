import pandas as pd
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# MySQL connection
conn = mysql.connector.connect(
    host="",
    user="",
    password="",
    database=""
)
cursor = conn.cursor()

# Fetch data from the database
cursor.execute("SELECT content, sentiment FROM review WHERE sentiment IS NOT NULL")
data = cursor.fetchall()

# Convert fetched data into a DataFrame
df = pd.DataFrame(data, columns=["content", "sentiment"])

# TF-IDF vectorization for text features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['content'])
y = df['sentiment']

# Train the logistic regression model
model = LogisticRegression(max_iter=1000, multi_class='multinomial')
model.fit(X, y)

# Save the trained model and vectorizer
joblib.dump(model, 'review_sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Evaluate the model performance
y_pred = model.predict(X)
print(classification_report(y, y_pred))