import pandas as pd
import numpy as np

# Number of rows to generate
n = 100

# Generate synthetic data
df = pd.DataFrame({
    "loan_id": np.arange(1, n+1),
    "no_of_dependents": np.random.randint(0, 4, size=n),
    "education": np.random.choice(["Graduate", "Not Graduate"], size=n),
    "self_employed": np.random.choice(["Yes", "No"], size=n),
    "income_annum": np.random.randint(100000, 1000000, size=n),
    "loan_amount": np.random.randint(50000, 500000, size=n),
    "loan_term": np.random.choice([12, 24, 36, 60, 120], size=n),
    "cibil_score": np.random.randint(300, 900, size=n),
    "residential_assets_value": np.random.randint(100000, 1000000, size=n),
    "commercial_assets_value": np.random.randint(50000, 500000, size=n),
    "luxury_assets_value": np.random.randint(20000, 200000, size=n),
    "bank_asset_value": np.random.randint(50000, 500000, size=n),
    " loan_status": np.random.choice(["Approved", "Rejected"], size=n)  # note leading space to match your code
})

# Save to CSV
df.to_csv("test_data.csv", index=False)

print("Synthetic test data generated and saved as test_data.csv")
print(df.head())
