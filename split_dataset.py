"""
STEAD Dataset Splitting and Validation Script

This script splits a large seismic dataset into multiple splits, validates the splits,
plots distributions, and writes the splits to disk. It is designed to be flexible for any
number of splits as defined in the SPLIT_SIZES constant.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Constants
SPLIT_SIZES = [100000, 40000]  # Example: [100000, 40000] for 3 splits (last is 'rest')
OUTPUT_DIR = (
    'C:/Users/cadas/Box/ECSTATIC - Private/STEAD-chunks/'
)


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
    """Calculate and print the proportion of earthquakes and noise in datasets for variable splits."""
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
    for i, split in enumerate(splits):
        eq = split[split['trace_category'] == 'earthquake_local']
        noise = split[split['trace_category'] == 'noise']
        eq_splits.append(eq)
        noise_splits.append(noise)
        eq_prop = len(eq) / len(split) if len(split) > 0 else 0
        noise_prop = len(noise) / len(split) if len(split) > 0 else 0
        print(f"Split {i+1}: Earthquakes {len(eq)} ({eq_prop:.2%}, diff: {eq_prop-orig_eq_prop:+.2%}), "
              f"Noise {len(noise)} ({noise_prop:.2%}, diff: {noise_prop-orig_noise_prop:+.2%})")
    return eq_splits, noise_splits


def validate_splits(df_orig, splits, file_name, output_dir):
    """Validate the statistical similarity of the splits (variable number of splits)."""
    print(f"\nValidating splits for {file_name}:")
    numerical_columns = splits[0].select_dtypes(include=[np.number]).columns
    sns.color_palette("colorblind")
    eq_splits, noise_splits = calculate_proportion(df_orig, splits)
    for column in numerical_columns:
        print(f"\n ----------------- Analyzing {column}: --------------------")
        plt.figure(figsize=(10, 6))
        data = [df_orig[column].dropna()] + [split[column].dropna() for split in splits]
        labels = ['Original'] + [f'Split {i+1}' for i in range(len(splits))]
        plt.boxplot(data, tick_labels=labels)
        title = f'Boxplot of {column}'
        plt.title(title)
        plt.ylabel(column)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        boxplot_path = os.path.join(output_dir, f'{file_name}_{column}_boxplot.png')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(boxplot_path)
        plt.close()


def plot_seismograms_per_year(df, file_name, output_dir):
    """Create a bar plot showing the number of seismograms per year."""
    df = df.copy()
    df['year'] = pd.to_datetime(df['source_origin_time']).dt.year
    yearly_counts = df['year'].value_counts().sort_index()
    plt.figure(figsize=(15, 6))
    bars = plt.bar(yearly_counts.index, yearly_counts.values, color=sns.color_palette("colorblind")[0])
    plt.title(f'Number of Seismograms per Year - {file_name}')
    plt.xlabel('Year')
    plt.ylabel('Number of Seismograms')
    plt.xticks(rotation=45)
    plt.tight_layout()
    yearly_plot_path = os.path.join(output_dir, f'{file_name}_yearly_distribution.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(yearly_plot_path)
    plt.close()


def plot_epicentral_distance_pie(df, file_name, output_dir):
    """Plot a pie chart of epicentral distances for earthquake_local events."""
    eq_df = df[df['trace_category'] == 'earthquake_local']
    bins = [0, 5, 10, 20, 40, 80, 110, 350]
    labels = ['0-5 km', '5-10 km', '10-20 km', '20-40 km', '40-80 km', '80-110 km', '110-350 km']
    categories = pd.cut(eq_df['source_distance_km'], bins=bins, labels=labels, right=False)
    counts = categories.value_counts(sort=False)
    plt.figure(figsize=(6, 6))
    wedges, texts, autotexts = plt.pie(
        counts, labels=labels, autopct='%1.0f %%', startangle=90,
        colors=plt.cm.winter(np.linspace(0, 1, len(labels)))
    )
    plt.title('Epicentral Distances', fontsize=16, fontweight='bold')
    plt.tight_layout()
    pie_path = os.path.join(output_dir, f'{file_name}_epicentral_distance_pie.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(pie_path)
    plt.close()


def plot_depth_distribution(df, file_name, output_dir):
    """Plot the distribution of earthquake depths (log frequency vs. depth in km)."""
    eq_df = df[df['trace_category'] == 'earthquake_local']
    depths = eq_df['source_depth_km'].dropna()
    plt.figure(figsize=(10, 6))
    plt.hist(depths, bins=35, color='cornflowerblue', alpha=0.7)
    plt.yscale('log')
    plt.xlabel('Depth km')
    plt.ylabel('Log Frequency')
    plt.title('Distribution of earthquake depths')
    plt.tight_layout()
    depth_plot_path = os.path.join(output_dir, f'{file_name}_depth_distribution.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(depth_plot_path)
    plt.close()


def plot_magnitude_distribution(df, file_name, output_dir):
    """Plot the distribution of earthquake magnitudes (log frequency vs. magnitude)."""
    eq_df = df[df['trace_category'] == 'earthquake_local']
    mags = eq_df['source_magnitude'].dropna()
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(mags, bins=35, color='mediumpurple', alpha=0.7)
    plt.yscale('log')
    plt.xlabel('Magnitude')
    plt.ylabel('Log Frequency')
    plt.title('Distribution of earthquake magnitudes')
    if len(mags) > 0:
        max_mag = mags.max()
        min_mag = mags.min()
        plt.text(0.98, 0.98, f"Max: {max_mag:.2f} M\nMin: {min_mag:.2f} M", 
                 ha='right', va='top', transform=plt.gca().transAxes,
                 fontsize=12, bbox=dict(facecolor='lavender', alpha=0.7, edgecolor='none'))
    plt.tight_layout()
    mag_plot_path = os.path.join(output_dir, f'{file_name}_magnitude_distribution.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(mag_plot_path)
    plt.close()


def plot_all_depth_distributions(dfs, names, output_dir, base):
    """Plot all depth distributions as subplots in a single figure."""
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]
    for i, (df, name) in enumerate(zip(dfs, names)):
        eq_df = df[df['trace_category'] == 'earthquake_local']
        depths = eq_df['source_depth_km'].dropna()
        axes[i].hist(depths, bins=35, color='cornflowerblue', alpha=0.7)
        axes[i].set_yscale('log')
        axes[i].set_xlabel('Depth km')
        if i == 0:
            axes[i].set_ylabel('Log Frequency')
        axes[i].set_title(name)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{base}_all_depth_distributions.png'))
    plt.close()


def plot_all_seismograms_per_year(dfs, names, output_dir, base):
    """Plot all seismograms per year as subplots in a single figure."""
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]
    for i, (df, name) in enumerate(zip(dfs, names)):
        df = df.copy()
        df['year'] = pd.to_datetime(df['source_origin_time']).dt.year
        yearly_counts = df['year'].value_counts().sort_index()
        axes[i].bar(yearly_counts.index, yearly_counts.values, color=sns.color_palette("colorblind")[0])
        axes[i].set_title(name)
        axes[i].set_xlabel('Year')
        if i == 0:
            axes[i].set_ylabel('Number of Seismograms')
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{base}_all_seismograms_per_year.png'))
    plt.close()


def plot_all_epicentral_distance_pies(dfs, names, output_dir, base):
    """Plot all epicentral distance pies as subplots in a single figure."""
    bins = [0, 5, 10, 20, 40, 80, 110, 350]
    labels = ['0-5 km', '5-10 km', '10-20 km', '20-40 km', '40-80 km', '80-110 km', '110-350 km']
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    if n == 1:
        axes = [axes]
    for i, (df, name) in enumerate(zip(dfs, names)):
        eq_df = df[df['trace_category'] == 'earthquake_local']
        categories = pd.cut(eq_df['source_distance_km'], bins=bins, labels=labels, right=False)
        counts = categories.value_counts(sort=False)
        axes[i].pie(
            counts, labels=labels, autopct='%1.0f %%', startangle=90,
            colors=plt.cm.winter(np.linspace(0, 1, len(labels)))
        )
        axes[i].set_title(name)
    plt.suptitle('Epicentral Distances', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{base}_all_epicentral_distance_pies.png'))
    plt.close()


def plot_all_magnitude_distributions(dfs, names, output_dir, base):
    """Plot all magnitude distributions as subplots in a single figure."""
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]
    for i, (df, name) in enumerate(zip(dfs, names)):
        eq_df = df[df['trace_category'] == 'earthquake_local']
        mags = eq_df['source_magnitude'].dropna()
        axes[i].hist(mags, bins=35, color='mediumpurple', alpha=0.7)
        axes[i].set_yscale('log')
        axes[i].set_xlabel('Magnitude')
        if i == 0:
            axes[i].set_ylabel('Log Frequency')
        axes[i].set_title(name)
        if len(mags) > 0:
            max_mag = mags.max()
            min_mag = mags.min()
            axes[i].text(0.98, 0.98, f"Max: {max_mag:.2f} M\nMin: {min_mag:.2f} M", 
                         ha='right', va='top', transform=axes[i].transAxes,
                         fontsize=12, bbox=dict(facecolor='lavender', alpha=0.7, edgecolor='none'))
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{base}_all_magnitude_distributions.png'))
    plt.close()


def main():
    """Main function to load data, split, validate, plot, and write outputs."""
    parser = argparse.ArgumentParser(
        description=(
            "STEAD Dataset Splitting and Validation\n\n"
            "Example usage:\n"
            "  python split_dataset.py --input_files mydata.csv --output_dir ./splits --split_sizes 100000 40000"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--input_files',
        nargs='+',
        required=True,
        help='Path(s) to input CSV file(s) to process.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='C:/Users/cadas/Box/ECSTATIC - Private/STEAD-chunks/',
        help='Directory to save output split files.'
    )
    parser.add_argument(
        '--split_sizes',
        type=int,
        nargs='+',
        default=[100000, 40000],
        help='List of split sizes (last split will be the rest).'
    )
    args = parser.parse_args()
    input_files = args.input_files
    output_dir = args.output_dir
    split_sizes = args.split_sizes
    
    for file_path in input_files:
        # Load the original dataset
        df_orig = pd.read_csv(file_path, low_memory=False)
        print(f"{file_path}: {len(df_orig)} rows loaded")
        print("First 3 rows:\n", df_orig.head(3))
        print("Last 3 rows:\n", df_orig.tail(3))

        # Shuffle the dataset for random splitting
        df_shuf = df_orig.sample(frac=1, random_state=42).reset_index(drop=True)

        # Generate splits based on the provided split sizes
        splits = generate_splits(df_shuf, split_sizes)
        base = os.path.splitext(os.path.basename(file_path))[0]

        # Validate the splits and generate boxplots
        validate_splits(df_orig, splits, base, output_dir)

        # Plot yearly seismogram distribution for the original dataset
        plot_seismograms_per_year(df_orig, f'{base}_original', output_dir)
        
        # Plot epicentral distance pie chart for the original dataset
        plot_epicentral_distance_pie(df_orig, f'{base}_original', output_dir)

        # Plot depth distribution for the original dataset
        plot_depth_distribution(df_orig, f'{base}_original', output_dir)

        # Plot magnitude distribution for the original dataset
        plot_magnitude_distribution(df_orig, f'{base}_original', output_dir)

        # Plot yearly seismogram distribution for each split
        for i, split in enumerate(splits):
            plot_seismograms_per_year(split, f'{base}_split{i+1}', output_dir)
            
            plot_epicentral_distance_pie(split, f'{base}_split{i+1}', output_dir)

            # Plot depth distribution for each split
            plot_depth_distribution(split, f'{base}_split{i+1}', output_dir)

            # Plot magnitude distribution for each split
            plot_magnitude_distribution(split, f'{base}_split{i+1}', output_dir)

        # Prepare list of all DataFrames and their names
        all_dfs = [df_orig] + splits
        all_names = [f'{base}_original'] + [f'{base}_split{i+1}' for i in range(len(splits))]

        # Plot all seismograms per year in one figure
        plot_all_seismograms_per_year(all_dfs, all_names, output_dir, base)

        # Plot all epicentral distance pies in one figure
        plot_all_epicentral_distance_pies(all_dfs, all_names, output_dir, base)

        # Plot all depth distributions in one figure
        plot_all_depth_distributions(all_dfs, all_names, output_dir, base)

        # Plot all magnitude distributions in one figure
        plot_all_magnitude_distributions(all_dfs, all_names, output_dir, base)

        # Prepare output file paths for each split
        outfiles = [
            os.path.join(
                output_dir,
                f'{base}_' + (f'{size}k.csv' if i < len(split_sizes) else 'rest.csv')
            )
            for i, size in enumerate(split_sizes + [''])
        ]

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Write each split to its corresponding CSV file
        for split, outfile in zip(splits, outfiles):
            split.to_csv(outfile, index=False)


if __name__ == "__main__":
    main()
