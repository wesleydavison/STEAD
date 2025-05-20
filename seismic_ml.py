# seismic_ml.py
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, confusion_matrix, classification_report)
import joblib

def print_timestamp(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

class SeismicDataProcessor:
    """Class for reading and storing seismic data"""
    def __init__(self, file_name_eq, file_name_noise, csv_file_eq, csv_file_noise, 
                 mode='prod', eq_filters=None, nrows=None, chunksize=None):
        self.file_name_eq = file_name_eq
        self.file_name_noise = file_name_noise
        self.csv_file_eq = csv_file_eq
        self.csv_file_noise = csv_file_noise
        self.mode = mode
        self.eq_filters = eq_filters or {'trace_category': 'earthquake_local'}
        self.eq_data = {}
        self.noise_data = {}
        if mode == 'fast':
            self.chunksize = chunksize if chunksize is not None else 20000
            self.nrows = nrows if nrows is not None else 20000
        else:
            self.chunksize = None
            self.nrows = None
    def _load_data_chunks(self):
        needed_columns = ['trace_name', 'trace_category', 'source_magnitude', 'source_distance_km']
        dtype_dict = {
            'trace_name': 'category',
            'trace_category': 'category',
            'source_magnitude': 'float32',
            'source_distance_km': 'float32'
        }
        if self.mode == 'fast':
            if self.nrows is not None and self.nrows <= self.chunksize:
                chunks_eq = [pd.read_csv(self.csv_file_eq, nrows=self.nrows, usecols=needed_columns, dtype=dtype_dict, low_memory=False)]
                chunks_noise = [pd.read_csv(self.csv_file_noise, nrows=self.nrows, usecols=needed_columns, dtype=dtype_dict)]
                total_chunks = 1
            else:
                chunks_eq = pd.read_csv(self.csv_file_eq, chunksize=self.chunksize, nrows=self.nrows, 
                                      usecols=needed_columns, dtype=dtype_dict, low_memory=False)
                chunks_noise = pd.read_csv(self.csv_file_noise, chunksize=self.chunksize, nrows=self.nrows, 
                                         usecols=needed_columns, dtype=dtype_dict)
                total_chunks = min(self.nrows // self.chunksize + (1 if self.nrows % self.chunksize else 0), 
                                 len(list(pd.read_csv(self.csv_file_eq, chunksize=self.chunksize, nrows=self.nrows))))
        else:
            chunks_eq = [pd.read_csv(self.csv_file_eq, usecols=needed_columns, dtype=dtype_dict)]
            chunks_noise = [pd.read_csv(self.csv_file_noise, usecols=needed_columns, dtype=dtype_dict)]
            total_chunks = 1
        return chunks_eq, chunks_noise, total_chunks
    def _filter_data(self, chunk_eq, chunk_noise):
        if self.mode == 'fast':
            filtered_eq = chunk_eq[chunk_eq.trace_category == self.eq_filters['trace_category']]
            if 'source_distance_km' in self.eq_filters and self.eq_filters['source_distance_km'] is not None:
                filtered_eq = filtered_eq[filtered_eq.source_distance_km <= self.eq_filters['source_distance_km']]
            if 'source_magnitude' in self.eq_filters and self.eq_filters['source_magnitude'] is not None:
                mag_min, mag_max = self.eq_filters['source_magnitude']
                filtered_eq = filtered_eq[
                    (filtered_eq.source_magnitude > mag_min) & 
                    (filtered_eq.source_magnitude < mag_max)
                ]
        else:
            filtered_eq = chunk_eq[chunk_eq.trace_category == self.eq_filters['trace_category']]
        filtered_noise = chunk_noise[chunk_noise.trace_category == 'noise']
        return filtered_eq, filtered_noise, filtered_noise
    def load_data(self):
        chunks_eq, chunks_noise, total_chunks = self._load_data_chunks()
        with tqdm(total=total_chunks, desc=f"Loading chunks ({self.mode} mode)") as pbar:
            for chunk_eq, chunk_noise in zip(chunks_eq, chunks_noise):
                filtered_eq, filtered_noise, _ = self._filter_data(chunk_eq, chunk_noise)
                if filtered_eq.empty:
                    print("No events found in this chunk")
                    pbar.update(1)
                    continue
                ev_list_eq = filtered_eq['trace_name'].tolist()
                ev_list_noise = filtered_noise['trace_name'].tolist()
                with h5py.File(self.file_name_eq, 'r') as dtfl_eq, h5py.File(self.file_name_noise, 'r') as dtfl_noise:
                    for trace_name_eq in tqdm(ev_list_eq, desc="Earthquake traces", leave=False):
                        ds_eq = dtfl_eq.get(f"data/{trace_name_eq}")
                        if ds_eq is None:
                            continue
                        data_eq = np.array(ds_eq)
                        fs_eq = float(ds_eq.attrs.get('sampling_rate', 100.0))
                        mag = filtered_eq[filtered_eq.trace_name == trace_name_eq].source_magnitude.iloc[0]
                        self.eq_data[trace_name_eq] = (data_eq, fs_eq, mag)
                    for trace_name_noise in tqdm(ev_list_noise, desc="Noise traces", leave=False):
                        ds_noise = dtfl_noise.get(f"data/{trace_name_noise}")
                        if ds_noise is None:
                            continue
                        data_noise = np.array(ds_noise)
                        fs_noise = float(ds_noise.attrs.get('sampling_rate', 100.0))
                        self.noise_data[trace_name_noise] = (data_noise, fs_noise)
                pbar.update(1)
        print(f"\nLoaded data:")
        print(f"- {len(self.eq_data)} earthquake traces")
        print(f"- {len(self.noise_data)} noise traces")
        return self.eq_data, self.noise_data

class SeismicClassifier:
    """Class for seismic signal classification using a pipeline."""
    def __init__(self, test_size=0.3, random_state=42, n_bins=50, feature_transformer=None):
        self.test_size = test_size
        self.random_state = random_state
        self.n_bins = n_bins
        self.pipeline = None
        self.training_time = None
        if feature_transformer is not None:
            self.feature_transformer = feature_transformer
        else:
            from psd_classification import PSDFeatureTransformer
            self.feature_transformer = PSDFeatureTransformer(n_bins=self.n_bins)
    def _prepare_data(self, eq_data, noise_data):
        X = []
        y = []
        trace_names = []
        for k, v in eq_data.items():
            X.append((v[0], v[1]))
            y.append(1)
            trace_names.append(k)
        for k, v in noise_data.items():
            X.append((v[0], v[1]))
            y.append(0)
            trace_names.append(k)
        return np.array(X, dtype=object), np.array(y), np.array(trace_names)
    def train(self, eq_data_train, noise_data_train, eq_data_val, noise_data_val):
        print_timestamp("Starting training process...")
        X_train, y_train, trace_names_train = self._prepare_data(eq_data_train, noise_data_train)
        X_val, y_val, trace_names_val = self._prepare_data(eq_data_val, noise_data_val)
        overlap = set(trace_names_train) & set(trace_names_val)
        print_timestamp(f"Number of overlapping trace names between train and validation: {len(overlap)}")
        if overlap:
            print_timestamp(f"Example overlaps: {list(overlap)[:10]}")
        else:
            print_timestamp("No overlap detected!")
        print_timestamp(f"Train set: {len(X_train)} samples ({sum(y_train)} earthquakes, {len(y_train)-sum(y_train)} noise)")
        print_timestamp(f"Validation set: {len(X_val)} samples ({sum(y_val)} earthquakes, {len(y_val)-sum(y_val)} noise)")
        self.pipeline = Pipeline([
            ('feature', self.feature_transformer),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=1000,
                solver='saga',
                verbose=1,
                C=0.01,
                class_weight='balanced',
                n_jobs=-1,
                random_state=self.random_state,
                tol=1e-4,
                penalty='elasticnet',
                l1_ratio=0.5
            ))
        ])
        print_timestamp("Extracting features...")
        feature_extractor = self.pipeline.named_steps['feature']
        X_train_features = feature_extractor.transform(X_train)
        if hasattr(feature_extractor, 'valid_indices_') and feature_extractor.valid_indices_ is not None:
            y_train = y_train[feature_extractor.valid_indices_]
            X_train = X_train[feature_extractor.valid_indices_]
        X_val_features = feature_extractor.transform(X_val)
        if hasattr(feature_extractor, 'valid_indices_') and feature_extractor.valid_indices_ is not None:
            y_val = y_val[feature_extractor.valid_indices_]
            X_val = X_val[feature_extractor.valid_indices_]
        np.save('train_features.npy', X_train_features)
        np.save('val_features.npy', X_val_features)
        np.save('train_labels.npy', y_train)
        np.save('val_labels.npy', y_val)
        print_timestamp("Training...")
        import time
        start_time = time.time()
        self.pipeline.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        print_timestamp("Evaluating...")
        y_val_pred = self.pipeline.predict(X_val)
        y_pred_proba = self.pipeline.predict_proba(X_val)[:, 1]
        metrics = self.evaluate(y_val, y_val_pred, y_pred_proba)
        if hasattr(self.pipeline['clf'], 'coef_'):
            coef = self.pipeline['clf'].coef_[0]
            print_timestamp("Top 10 most important features:")
            for idx in np.argsort(np.abs(coef))[-10:]:
                print_timestamp(f"  Feature {idx}: {coef[idx]:.4f}")
        return metrics
    def predict(self, X):
        if self.pipeline is None:
            raise ValueError("Pipeline must be trained or loaded before prediction.")
        return self.pipeline.predict(X)
    def predict_proba(self, X):
        if self.pipeline is None:
            raise ValueError("Pipeline must be trained or loaded before prediction.")
        return self.pipeline.predict_proba(X)
    def save_pipeline(self, path):
        if self.pipeline is None:
            raise ValueError("No trained pipeline to save")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
        print_timestamp(f"Pipeline saved to {path}")
    def load_pipeline(self, path):
        self.pipeline = joblib.load(path)
        print_timestamp(f"Pipeline loaded from {path}")
    @staticmethod
    def evaluate(y_true, y_pred, y_pred_proba):
        results = {}
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred)
        results['recall'] = recall_score(y_true, y_pred)
        results['f1'] = f1_score(y_true, y_pred)
        results['pr_auc'] = average_precision_score(y_true, y_pred_proba)
        results['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        results['classification_report'] = classification_report(y_true, y_pred, digits=3)
        return results 