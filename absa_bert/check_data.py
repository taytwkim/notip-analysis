import pandas as pd

df = pd.read_csv("absa_train_dataset.csv")

invalid_labels = df[['good_ambience', 'good_service', 'good_taste', 'good_price']].applymap(lambda x: x not in [0, 1])

if invalid_labels.any().any():
    print("Invalid labels found:")
    print(df[invalid_labels.any(axis=1)])
else:
    print("All labels are valid.")
