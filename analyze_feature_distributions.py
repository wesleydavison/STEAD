import numpy as np
import matplotlib.pyplot as plt

# Load features
train = np.load('train_features.npy')
val = np.load('val_features.npy')
test = np.load('test_features.npy')

print(f"Train features shape: {train.shape}")
print(f"Val features shape: {val.shape}")
print(f"Test features shape: {test.shape}")

def print_stats(name, arr):
    print(f"{name}:")
    print(f"  mean: {np.mean(arr, axis=0)[:5]}")
    print(f"  std:  {np.std(arr, axis=0)[:5]}")
    print(f"  min:  {np.min(arr, axis=0)[:5]}")
    print(f"  max:  {np.max(arr, axis=0)[:5]}")

print_stats("Train", train)
print_stats("Val", val)
print_stats("Test", test)

# Plot histograms for the first 5 features
for i in range(5):
    plt.figure()
    plt.hist(train[:, i], bins=30, alpha=0.5, label='Train')
    plt.hist(val[:, i], bins=30, alpha=0.5, label='Val')
    plt.hist(test[:, i], bins=30, alpha=0.5, label='Test')
    plt.title(f'Feature {i}')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'feature_{i}_hist.png')  # Save instead of show for non-GUI
    plt.close() 