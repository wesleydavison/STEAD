import numpy as np
import pandas as pd
import h5py
import os
from PyEMD import EMD
import matplotlib.pyplot as plt

def plot_emd_features(data, fs, imfs, n_imfs, features=None, title="EMD Features"):
    """Plot EMD features for a single trace"""
    n_imfs_to_plot = min(n_imfs, max(imf.shape[0] for imf in imfs))
    n_features_per_comp = n_imfs_to_plot * 2 + 1  # energies + frequencies + error
    fig = plt.figure(figsize=(15, 4*(n_imfs_to_plot + 2)))
    gs = fig.add_gridspec(n_imfs_to_plot + 2, 3)
    fig.suptitle(title)
    t = np.arange(len(data)) / fs
    components = ['e', 'n', 'z']
    for comp_idx, comp in enumerate(components):
        # Original signal
        ax = fig.add_subplot(gs[0, comp_idx])
        ax.plot(t, data[:, comp_idx], 'k', lw=0.5)
        ax.set_title(f'{comp} Component')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        # IMFs
        for i in range(n_imfs_to_plot):
            if i < imfs[comp_idx].shape[0]:
                ax = fig.add_subplot(gs[i+1, comp_idx])
                ax.plot(t, imfs[comp_idx][i], 'b', lw=0.5)
                ax.set_title(f'IMF {i+1}')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"emd_features_{title}.png")
    return fig

def decompose_and_save_emd(
    file_name_hdf5, 
    csv_file, 
    output_file, 
    n_samples=10, 
    components=['e', 'n', 'z'],
    plot_random_traces=5
):
    print(f"Processing: {csv_file}")
    dtype_dict = {
        'trace_name': 'category',
        'trace_category': 'category',
        'source_magnitude': 'float32'
    }
    df = pd.read_csv(csv_file, usecols=['trace_name', 'trace_category', 'source_magnitude'], dtype=dtype_dict)
    all_indices = np.arange(len(df))
    np.random.seed(42)
    sampled_indices = np.random.choice(all_indices, size=min(n_samples, len(df)), replace=False)
    sampled_df = df.iloc[sampled_indices]
    eq_data = {}
    noise_data = {}
    with h5py.File(output_file, 'w') as f_out:
        f_out.attrs['creation_date'] = np.string_(pd.Timestamp.now().isoformat())
        f_out.attrs['n_samples'] = n_samples
        with h5py.File(file_name_hdf5, 'r') as h5f:
            for _, row in sampled_df.iterrows():
                trace_name = row['trace_name']
                trace_category = row['trace_category']
                mag = row.get('source_magnitude', 0)
                ds = h5f.get(f"data/{trace_name}")
                if ds is None:
                    continue
                data = np.array(ds)
                fs = float(ds.attrs.get('sampling_rate', 100.0))
                if trace_category == 'earthquake_local':
                    eq_data[trace_name] = (data, fs, mag)
                else:
                    noise_data[trace_name] = (data, fs)
                # EMD decomposition for each component
                imfs = []
                for comp_idx in range(data.shape[1]):
                    emd = EMD()
                    imfs_comp = emd.emd(data[:, comp_idx])
                    imfs.append(imfs_comp)
                plot_emd_features(data, fs, imfs, n_imfs=10, title=f"EMD_Decomposition_{trace_name}")
                # Save decomposition results
                trace_group = f_out.create_group(trace_name)
                trace_group.attrs['sampling_rate'] = fs
                trace_group.attrs['magnitude'] = mag
                trace_group.attrs['category'] = trace_category
                trace_group.create_dataset('original_data', data=data)
                for comp_idx, (comp_name, imf_data) in enumerate(zip(components, imfs)):
                    comp_group = trace_group.create_group(f'component_{comp_name}')
                    comp_group.create_dataset('imfs', data=imf_data)
                print(f"Saved decomposition for trace: {trace_name}")
        print(f"All decompositions saved to: {output_file}")
        # Optionally plot random traces
        print("Checking if the file saved correctly...")
        print(f"File contains the following groups: {list(f_out.keys())}")
        for group in f_out.keys():
            print(f"Group {group} contains the following datasets: {list(f_out[group].keys())}")
        print("Choosing random traces to plot...")
        random_trace_names = np.random.choice(list(f_out.keys()), size=min(plot_random_traces, len(f_out.keys())), replace=False)
        print(f"Plotting EMD decompositions for the following traces: {random_trace_names}")
        for trace_name in random_trace_names:
            trace_group = f_out[trace_name]
            imfs = []
            for comp in components:
                imfs.append(trace_group[f'component_{comp}']['imfs'][:])
            plot_emd_features(trace_group['original_data'][:], 
                              trace_group.attrs['sampling_rate'], 
                              imfs, 
                              n_imfs=10, 
                              title=f"EMD_test_Decomposition-{trace_name}")

# --- Configuration for earthquake and noise data ---
file_name_merged = r"/users/230442014/archive/STEAD_dataset/chunk2.hdf5"
csv_file_merged = r"/users/230442014/archive/STEAD_dataset/chunk2.csv"
output_file_merged = 'emd_decompositions_chunk2.h5'

file_name_noise = r"/users/230442014/archive/STEAD_dataset/chunk1.hdf5"
csv_file_noise = r"/users/230442014/archive/STEAD_dataset/chunk1.csv"
output_file_noise = 'emd_decompositions_chunk1.h5'

# Ensure model directory exists
os.makedirs("saved_models", exist_ok=True)

# Run for earthquake data
decompose_and_save_emd(
    file_name_hdf5=file_name_merged,
    csv_file=csv_file_merged,
    output_file=output_file_merged,
    n_samples=3
)

# Run for noise data
decompose_and_save_emd(
    file_name_hdf5=file_name_noise,
    csv_file=csv_file_noise,
    output_file=output_file_noise,
    n_samples=3
)





