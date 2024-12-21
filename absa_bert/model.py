from transformers import BertForSequenceClassification

def get_model():
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=4, problem_type="multi_label_classification"
    )
    return model
