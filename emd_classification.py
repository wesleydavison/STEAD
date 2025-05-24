# emd_classification.py
import time
from datetime import datetime
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           average_precision_score, roc_auc_score)
import joblib
import os
import argparse
from decompose_emd import decompose_and_save_emd

def print_timestamp(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seismic EMD Classification")
    parser.add_argument('--mode', type=str, default='prod', choices=['fast', 'prod'], help='Mode: test or prod')
    parser.add_argument('--nrows', type=int, default=20000, help='Number of rows to load in test mode')
    parser.add_argument('--chunksize', type=int, default=20000, help='Chunk size to use in test mode')
    # Add arguments for train/val file paths
    parser.add_argument('--eq_h5', type=str, required=True, help='Earthquake HDF5 file (shared)')
    parser.add_argument('--noise_h5', type=str, required=True, help='Noise HDF5 file (shared)')
    parser.add_argument('--train_eq_csv', type=str, required=True, help='Training earthquake CSV file')
    parser.add_argument('--train_noise_csv', type=str, required=True, help='Training noise CSV file')
    parser.add_argument('--val_eq_csv', type=str, required=True, help='Validation earthquake CSV file')
    parser.add_argument('--val_noise_csv', type=str, required=True, help='Validation noise CSV file')
    args = parser.parse_args()

    # First, use decompose_emd.py to process the data
    print_timestamp("Processing data with decompose_emd.py...")
    process_start = time.time()
    
    # Process training data
    train_eq_output = "emd_decompositions_train_eq.h5"
    train_noise_output = "emd_decompositions_train_noise.h5"
    
    # Get features directly from decompose_emd.py
    X_train_eq, y_train_eq, mag_train_eq = decompose_and_save_emd(
        file_name_hdf5=args.eq_h5,
        csv_file=args.train_eq_csv,
        output_file=train_eq_output,
        n_samples=args.nrows if args.mode == 'fast' else None,
        generate_plots=False,
        generate_boxplots=False,
        return_features=True  # This will return features instead of saving to file
    )
    
    X_train_noise, y_train_noise, mag_train_noise = decompose_and_save_emd(
        file_name_hdf5=args.noise_h5,
        csv_file=args.train_noise_csv,
        output_file=train_noise_output,
        n_samples=args.nrows if args.mode == 'fast' else None,
        generate_plots=False,
        generate_boxplots=False,
        return_features=True
    )
    
    # Process validation data
    X_val_eq, y_val_eq, mag_val_eq = decompose_and_save_emd(
        file_name_hdf5=args.eq_h5,
        csv_file=args.val_eq_csv,
        output_file="emd_decompositions_val_eq.h5",
        n_samples=args.nrows if args.mode == 'fast' else None,
        generate_plots=False,
        generate_boxplots=False,
        return_features=True
    )
    
    X_val_noise, y_val_noise, mag_val_noise = decompose_and_save_emd(
        file_name_hdf5=args.noise_h5,
        csv_file=args.val_noise_csv,
        output_file="emd_decompositions_val_noise.h5",
        n_samples=args.nrows if args.mode == 'fast' else None,
        generate_plots=False,
        generate_boxplots=False,
        return_features=True
    )
    
    process_end = time.time()
    print_timestamp(f"Data processing took {process_end - process_start:.2f} seconds.")
    
    # Combine training and validation data
    X_train = np.vstack([X_train_eq, X_train_noise])
    y_train = np.concatenate([y_train_eq, y_train_noise])
    X_val = np.vstack([X_val_eq, X_val_noise])
    y_val = np.concatenate([y_val_eq, y_val_noise])
    
    # Train classifier
    print_timestamp("Training classifier...")
    clf_start = time.time()
    
    # Create and train pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_val)
    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'auc_roc': roc_auc_score(y_val, y_pred_proba),
        'avg_precision': average_precision_score(y_val, y_pred_proba)
    }
    
    clf_end = time.time()
    print_timestamp(f"Classification took {clf_end - clf_start:.2f} seconds.")
    
    # Print results
    print_timestamp("\nClassification Results:")
    for metric, value in metrics.items():
        print_timestamp(f"  {metric}: {value:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_size = "all" if args.nrows is None else f"n{args.nrows}"
    MODEL_DIR = "saved_models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{timestamp}_{dataset_size}_emd_classifier.pkl")
    
    joblib.dump(pipeline, model_path)
    print_timestamp(f"Model saved to: {model_path}")
    
    total_time = time.time() - process_start
    print_timestamp(f"Total execution time: {total_time:.2f} seconds.") 