import mysql.connector
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

conn = mysql.connector.connect(
    host="",
    user="",
    password="",
    database=""
)
cursor = conn.cursor()

tokenizer = BertTokenizer.from_pretrained("./absa_model")
model = BertForSequenceClassification.from_pretrained("./absa_model", num_labels=4)
model.eval()

cursor.execute("""
    SELECT no, content FROM review
    WHERE good_ambience IS NULL
""")
reviews = cursor.fetchall()

for review in reviews:
    review_no = review[0]
    review_content = review[1]

    inputs = tokenizer(
        review_content,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    probabilities = [float(p) for p in probabilities]
    classifications = [1 if p >= 0.4 else 0 for p in probabilities]

    cursor.execute("""
        UPDATE review
        SET good_ambience = %s, good_service = %s, good_taste = %s, good_price = %s
        WHERE no = %s
    """, (classifications[0], classifications[1], classifications[2], classifications[3], review_no))

    cursor.execute("""
        INSERT INTO review_analysis_absa (review_no, good_ambience, good_service, good_taste, good_price)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        good_ambience = %s, good_service = %s, good_taste = %s, good_price = %s
    """, (review_no, probabilities[0], probabilities[1], probabilities[2], probabilities[3],
          probabilities[0], probabilities[1], probabilities[2], probabilities[3]))
    
    conn.commit()

print("Database updates completed.")

cursor.close()
conn.close()
