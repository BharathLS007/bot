import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
df = pd.read_csv('Medical_dataset/Training.csv')  # Full path to the uploaded file

# Step 2: Split the dataset into train and test (keeping all columns)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Step 3: Save the train and test datasets
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
