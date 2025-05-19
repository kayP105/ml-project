import pandas as pd
import os

# Verify files exist
if not all([os.path.exists(f) for f in ["Fake.csv", "True.csv"]]):
    raise FileNotFoundError("Missing Fake.csv or True.csv in current directory")

# Load datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake['label'] = 1  # 1 for fake
real['label'] = 0  # 0 for real

# Combine and save
combined = pd.concat([fake, real])
os.makedirs("data", exist_ok=True)  # Create data folder if missing
combined.to_csv("data/fake_news_dataset.csv", index=False)

print("✅ Dataset created at: data/fake_news_dataset.csv")