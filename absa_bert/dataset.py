import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch

class ABSADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {**{k: v[idx] for k, v in self.encodings.items()}, 'labels': self.labels[idx]}

def prepare_data(file_path):
    df = pd.read_csv(file_path)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['review'], df[['good_ambience', 'good_service', 'good_taste', 'good_price']], test_size=0.2, random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained("./absa_model")
    
    def tokenize(texts): return tokenizer(texts.tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt")
    
    train_encodings, val_encodings = tokenize(train_texts), tokenize(val_texts)
    train_dataset = ABSADataset(train_encodings, train_labels)
    val_dataset = ABSADataset(val_encodings, val_labels)
    return train_dataset, val_dataset
