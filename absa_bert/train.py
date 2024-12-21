import torch
from transformers import AdamW
from dataset import prepare_data
from model import get_model
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

train_dataset, val_dataset = prepare_data("absa_train_dataset.csv")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = get_model().to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = BCEWithLogitsLoss()

epochs = 3
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device, dtype=torch.float)
        loss = loss_fn(model(**inputs).logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / len(train_loader):.4f}")

# Save
model.save_pretrained("./absa_model", safe_serialization=True)
