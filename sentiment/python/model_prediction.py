import mysql.connector
import joblib
import numpy as np

# MySQL connection
conn = mysql.connector.connect(
    host="",
    user="",
    password="",
    database=""
)
cursor = conn.cursor()

# Load the trained model and vectorizer
model = joblib.load('review_sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Fetch reviews that do not have sentiment labels
cursor.execute("SELECT no, content FROM review WHERE sentiment IS NULL")
reviews = cursor.fetchall()

# Predict sentiment probabilities for each review and update the database
for review in reviews:
    review_no = review[0]
    review_content = review[1]
    review_vectorized = vectorizer.transform([review_content])

    # Get the prediction probabilities for each class
    proba = model.predict_proba(review_vectorized)[0]

    # The probabilities for 'good', 'neutral', and 'bad'
    sentiment_good = proba[2]  # Assuming 2 represents the "positive" class (good)
    sentiment_neutral = proba[1]  # Assuming 1 represents the "neutral" class
    sentiment_bad = proba[0]  # Assuming 0 represents the "negative" class (bad)

    # Update the sentiment analysis data in the review_analysis table
    cursor.execute("""
        INSERT INTO review_analysis (review_no, sentiment_good, sentiment_neutral, sentiment_bad)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        sentiment_good = %s, sentiment_neutral = %s, sentiment_bad = %s
    """, (review_no, sentiment_good, sentiment_neutral, sentiment_bad,
          sentiment_good, sentiment_neutral, sentiment_bad))
    conn.commit()

    # Update the sentiment field in the review table
    # Set sentiment based on the highest probability
    sentiment = int(np.argmax(proba) - 1)  # Convert to 1 (positive), 0 (neutral), or -1 (negative)
    cursor.execute("""
        UPDATE review
        SET sentiment = %s
        WHERE no = %s
    """, (sentiment, review_no))
    conn.commit()

print("Sentiment analysis update completed.")