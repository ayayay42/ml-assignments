import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Load data
X_cat_train = pd.read_csv("A5_train_category.csv")
X_cat_test = pd.read_csv("A5_test_category.csv")

# Initialize encoder (per-column mapping)
encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",  # Assign -1 to unseen test categories
    unknown_value=-1,                    # Custom value for unknowns
    encoded_missing_value=-2             # Handle NaNs if present
)

# Fit ONLY on training data
encoder.fit(X_cat_train)

# Transform train/test sets
X_cat_train_encoded = encoder.transform(X_cat_train)
X_cat_test_encoded = encoder.transform(X_cat_test)

# Convert to DataFrame (preserve column names)
X_cat_train_encoded = pd.DataFrame(X_cat_train_encoded, columns=X_cat_train.columns)
X_cat_test_encoded = pd.DataFrame(X_cat_test_encoded, columns=X_cat_test.columns)


# Save to CSV for modeling
X_cat_train_encoded.to_csv("X_train_processed.csv", index=False)
X_cat_test_encoded.to_csv("X_test_processed.csv", index=False)