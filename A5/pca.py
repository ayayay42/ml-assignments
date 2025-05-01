# pca_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import gc

# Configuration
FLOAT_TRAIN_PATH = 'A5_train_float.csv'
FLOAT_TEST_PATH = 'A5_test_float.csv'
CAT_TRAIN_PATH = 'A5_train_category_encoded.csv'
CAT_TEST_PATH = 'A5_test_category_encoded.csv'
OUTPUT_TRAIN = 'A5_train_pca_processed.csv'
OUTPUT_TEST = 'A5_test_pca_processed.csv'
TARGET_COMPONENTS = 1500  # Target reduced columns
CHUNK_SIZE = 1500         # Must be ≥ TARGET_COMPONENTS

def process_dataset(float_path, cat_path, output_path, scaler, pca):
    """Process and save data in chunks"""
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Create empty output file with headers
    pca_cols = [f'PC_{i+1}' for i in range(TARGET_COMPONENTS)]
    cat_cols = pd.read_csv(CAT_TRAIN_PATH, nrows=0).columns.tolist()
    pd.DataFrame(columns=pca_cols + cat_cols).to_csv(output_path, index=False)
    
    # Process data in chunks
    float_reader = pd.read_csv(float_path, chunksize=CHUNK_SIZE)
    cat_reader = pd.read_csv(cat_path, chunksize=CHUNK_SIZE)
    
    for float_chunk, cat_chunk in zip(float_reader, cat_reader):
        # Scale and transform
        scaled = scaler.transform(float_chunk)
        pca_features = pca.transform(scaled)[:, :TARGET_COMPONENTS]
        
        # Combine features
        combined = pd.DataFrame(
            np.hstack([pca_features, cat_chunk.to_numpy()]),
            columns=pca_cols + cat_chunk.columns.tolist()
        )
        
        # Append to CSV
        combined.to_csv(output_path, mode='a', header=False, index=False)
        gc.collect()

if __name__ == '__main__':
    # 1. Load full training data for PCA fitting
    print("Loading training data for PCA fitting...")
    full_train = pd.read_csv(FLOAT_TRAIN_PATH, dtype=np.float32)
    
    # 2. Preprocess with StandardScaler
    print("Scaling data...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(full_train)
    
    # 3. Fit PCA with randomized solver
    print("Fitting PCA...")
    pca = PCA(n_components=TARGET_COMPONENTS, svd_solver='randomized')
    pca.fit(scaled_data)
    
    # 4. Process datasets
    print("Processing training data...")
    process_dataset(FLOAT_TRAIN_PATH, CAT_TRAIN_PATH, OUTPUT_TRAIN, scaler, pca)
    
    print("Processing test data...")
    process_dataset(FLOAT_TEST_PATH, CAT_TEST_PATH, OUTPUT_TEST, scaler, pca)
    
    print(f"Success! Reduced from {full_train.shape[1]} to {TARGET_COMPONENTS} float columns.")
    print(f"Explained variance: {np.sum(pca.explained_variance_ratio_):.1%}")