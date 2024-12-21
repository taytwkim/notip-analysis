from transformers import BertTokenizer, BertForSequenceClassification
import torch

def predict(review):
    tokenizer = BertTokenizer.from_pretrained("./absa_model")
    model = BertForSequenceClassification.from_pretrained("./absa_model")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    print(inputs)

    outputs = model(**inputs)
    logits = outputs.logits

    print("Logits:", logits)

    predictions = torch.sigmoid(logits.clamp(min=-100, max=100)).cpu().detach().numpy()
    return predictions

review = "The ambience and taste were fantastic, but service was slow."
print(predict(review))
