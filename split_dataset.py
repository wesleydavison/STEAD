import pandas as pd

# List of input files
input_files = [
    '/users/230442014/archive/STEAD_dataset/chunk1.csv',
    '/users/230442014/archive/STEAD_dataset/chunk2.csv'
]

# Split sizes
split_sizes = [100000, 40000]  # df1: 100k, df2: 40k, df3: rest

for file_path in input_files:
    # Load the data
    df = pd.read_csv(file_path, low_memory=False)
    
    # Shuffle (set seed for reproducibility)
    df_shuf = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Compute split indices
    n1 = split_sizes[0]
    n2 = split_sizes[1]
    n = len(df_shuf)
    n3 = n - n1 - n2  # remaining
    
    # Slice
    df1 = df_shuf.iloc[:n1]
    df2 = df_shuf.iloc[n1:n1+n2]
    df3 = df_shuf.iloc[n1+n2:]
    
    # Output filenames
    base = file_path.split('/')[-1].replace('.csv', '')
    out1 = f'/users/230442014/archive/STEAD_dataset/{base}_100k.csv'
    out2 = f'/users/230442014/archive/STEAD_dataset/{base}_40k.csv'
    out3 = f'/users/230442014/archive/STEAD_dataset/{base}_rest.csv'
    
    # Write out
    df1.to_csv(out1, index=False)
    df2.to_csv(out2, index=False)
    df3.to_csv(out3, index=False)
