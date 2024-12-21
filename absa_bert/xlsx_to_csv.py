import pandas as pd
from sklearn.model_selection import train_test_split

xlsx_file = "absa_review_dataset.xlsx"
df = pd.read_excel(xlsx_file)

label_columns = ['good_ambience', 'good_service', 'good_taste', 'good_price']

df_cleaned = df.dropna(subset=label_columns)
df_cleaned = df_cleaned[df_cleaned[label_columns].applymap(lambda x: x in [0, 1]).all(axis=1)]  # Ensure all labels are 0 or 1

# Split dataset into train (550) and test (150)
train_df, test_df = train_test_split(df_cleaned, test_size=150, random_state=42)

train_csv_file = "absa_train_dataset.csv"
train_df.to_csv(train_csv_file, index=False)

test_csv_file = "absa_test_dataset.csv"
test_df.to_csv(test_csv_file, index=False)

print(f"Train dataset saved as: {train_csv_file}")
print(f"Test dataset saved as: {test_csv_file}")
