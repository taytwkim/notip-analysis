import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate
from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("./absa_model")
tokenizer = BertTokenizer.from_pretrained("./absa_model")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

test_csv_file = "absa_test_dataset.csv"
test_df = pd.read_csv(test_csv_file)

test_encodings = tokenizer(
    test_df['review'].tolist(),
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

test_labels = torch.tensor(test_df[['good_ambience', 'good_service', 'good_taste', 'good_price']].values, dtype=torch.float32)

test_encodings = {key: val.to(device) for key, val in test_encodings.items()}
test_labels = test_labels.to(device)

with torch.no_grad():
    outputs = model(**test_encodings)
    logits = outputs.logits
    predictions = torch.sigmoid(logits).cpu().numpy()

threshold = 0.4
binary_predictions = (predictions > threshold).astype(int)

precision, recall, f1, _ = precision_recall_fscore_support(
    test_labels.cpu().numpy(),
    binary_predictions,
    average=None
)

aspects = ['Good Ambience', 'Good Service', 'Good Taste', 'Good Price']
table = [[aspect, f"{precision[i]:.2f}", f"{recall[i]:.2f}", f"{f1[i]:.2f}"] for i, aspect in enumerate(aspects)]

precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    test_labels.cpu().numpy(),
    binary_predictions,
    average='macro'
)
table.append(["Macro Average", f"{precision_macro:.2f}", f"{recall_macro:.2f}", f"{f1_macro:.2f}"])

print(tabulate(table, headers=["Aspect", "Precision", "Recall", "F1 Score"], tablefmt="grid"))
