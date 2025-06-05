import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# List of input files
# input_files = [
#     "C:/Users/cadas/Box/ECSTATIC - Private/STEAD-chunks/chunk1.csv",
#     "C:/Users/cadas/Box/ECSTATIC - Private/STEAD-chunks/chunk2.csv"
# ]

input_files = [
    "C:/Users/cadas/Box/ECSTATIC - General/02 datasets (public)/STanford EArthquake Dataset (STEAD)/merge.csv"
]

# Split sizes
split_sizes = [100000, 40000]  # df1: 100k, df2: 40k, df3: rest

def generate_splits(df, split_sizes):
    """Generate splits from a DataFrame based on a list of split sizes. The last split is the rest."""
    splits = []
    start = 0
    for size in split_sizes:
        splits.append(df.iloc[start:start+size])
        start += size
    splits.append(df.iloc[start:])  # The rest
    return splits

def calculate_proportion(df_orig, splits):
    """Calculate and print the proportion of earthquakes and noise in datasets for a variable number of splits."""
    # Calculate proportions for original dataset
    orig_eq = len(df_orig[df_orig['trace_category'] == 'earthquake_local'])
    orig_noise = len(df_orig[df_orig['trace_category'] == 'noise'])
    orig_total = len(df_orig)
    orig_eq_prop = orig_eq / orig_total
    orig_noise_prop = orig_noise / orig_total
    print("\nOriginal dataset proportions:")
    print(f"Earthquakes: {orig_eq} ({orig_eq_prop:.2%})")
    print(f"Noise: {orig_noise} ({orig_noise_prop:.2%})")

    eq_splits = []
    noise_splits = []
    eq_props = []
    noise_props = []
    for i, split in enumerate(splits):
        eq = split[split['trace_category'] == 'earthquake_local']
        noise = split[split['trace_category'] == 'noise']
        eq_splits.append(eq)
        noise_splits.append(noise)
        eq_prop = len(eq) / len(split) if len(split) > 0 else 0
        noise_prop = len(noise) / len(split) if len(split) > 0 else 0
        eq_props.append(eq_prop)
        noise_props.append(noise_prop)
        print(f"Split {i+1}: Earthquakes {len(eq)} ({eq_prop:.2%}, diff: {eq_prop-orig_eq_prop:+.2%}), "
              f"Noise {len(noise)} ({noise_prop:.2%}, diff: {noise_prop-orig_noise_prop:+.2%})")
    return eq_splits, noise_splits

def validate_splits(df_orig, splits, file_name):
    """Validate the statistical similarity of the splits (variable number of splits)."""
    print(f"\nValidating splits for {file_name}:")
    
    # Get numerical columns
    numerical_columns = splits[0].select_dtypes(include=[np.number]).columns
   
    # Set colorblind-friendly palette
    colorblind_palette = sns.color_palette("colorblind")
   
    # Calculate proportions and get separated data
    eq_splits, noise_splits = calculate_proportion(df_orig, splits)
    
    # Compare each numerical column across splits
    for column in numerical_columns:
        print(f"\n ----------------- Analyzing {column}: --------------------")
       
        print(f"\nCategory distribution:")
        for i, (eq, noise) in enumerate(zip(eq_splits, noise_splits)):
            print(f"Split {i+1}: {len(eq)} earthquakes, {len(noise)} noise")
       
        # Create boxplots
        plt.figure(figsize=(10, 6))
        
        # Boxplot for all data, handling missing values
        data = [df_orig[column].dropna()] + [split[column].dropna() for split in splits]
        labels = ['Original'] + [f'Split {i+1}' for i in range(len(splits))]
        plt.boxplot(data, labels=labels)
       
      
        title = f'Boxplot of {column}'
        plt.title(title)
        plt.ylabel(column)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{file_name}_{column}_boxplot.png')
        plt.close()
       


def plot_seismograms_per_year(df, file_name):
    """Create a bar plot showing the number of seismograms per year"""
    # Convert source_origin_time to datetime
    df['year'] = pd.to_datetime(df['source_origin_time']).dt.year
    
    # Count seismograms per year
    yearly_counts = df['year'].value_counts().sort_index()
    
    # Create the plot
    plt.figure(figsize=(15, 6))
    # Use colorblind-friendly color
    bars = plt.bar(yearly_counts.index, yearly_counts.values, color=sns.color_palette("colorblind")[0])
    plt.title(f'Number of Seismograms per Year - {file_name}')
    plt.xlabel('Year')
    plt.ylabel('Number of Seismograms')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{file_name}_yearly_distribution.png')
    plt.close()

for file_path in input_files:
    # Load the data
    df_orig = pd.read_csv(file_path, low_memory=False)
    print(f"{file_path}: {len(df_orig)} rows loaded")
    print("First 3 rows:\n", df_orig.head(3))
    print("Last 3 rows:\n", df_orig.tail(3))
    # Shuffle (set seed for reproducibility)
    df_shuf = df_orig.sample(frac=1, random_state=42).reset_index(drop=True)
    # Generate splits using the helper function
    splits = generate_splits(df_shuf, split_sizes)
    # Get base name for output files
    base = file_path.split('/')[-1].replace('.csv', '')
    # Validate splits and create plots only for chunk2
    if 'merge' in file_path:
        validate_splits(df_orig, splits, base)
        for i, split in enumerate(splits):
            plot_seismograms_per_year(split, f'{base}_split{i+1}')
    # Output filenames
    outfiles = [f'C:/Users/cadas/Box/ECSTATIC - Private/STEAD-chunks/DEBUG_{base}_' +
                (f'{size}k.csv' if i < len(split_sizes) else 'rest.csv')
                for i, size in enumerate(split_sizes + [''])]
    # Write out
    for split, outfile in zip(splits, outfiles):
        split.to_csv(outfile, index=False)
