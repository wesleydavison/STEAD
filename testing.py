import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# %% Define file paths and chunk size
file_name = r"C:\Users\cadas\Box\ECSTATIC - General\02 datasets (public)\STanford EArthquake Dataset (STEAD)\chunk1.hdf5"
csv_file = r"C:\Users\cadas\Box\ECSTATIC - General\02 datasets (public)\STanford EArthquake Dataset (STEAD)\chunk1.csv"
chunksize = 10000
nrows=10000
plotting = False

# %% Initialize lists to store maximum amplitudes for each channel and corresponding source magnitudes
max_amplitudes_e = []
max_amplitudes_n = []
max_amplitudes_z = []
source_magnitudes = []

# %% Reading the CSV file in chunks and processing each event
chunks = pd.read_csv(csv_file, chunksize=chunksize, nrows=nrows)
for chunk in chunks:
    print(f'Total events in CSV chunk: {len(chunk)}')
    # Filter the dataframe according to your criteria.
    # chunk = chunk[(chunk.trace_category == 'earthquake_local') &
    #               (chunk.source_distance_km <= 20) &
    #               (chunk.source_magnitude > 3)]
    print(f'Total events selected in chunk: {len(chunk)}')

    # Make a list of trace names for the selected events.
    ev_list = chunk['trace_name'].to_list()

    # Open the HDF5 file containing the waveforms.
    dtfl = h5py.File(file_name, 'r')
    for c, evi in enumerate(ev_list):
        dataset = dtfl.get('data/' + str(evi))
        if dataset is None:
            print(f"Dataset for event {evi} not found, skipping.")
            continue
        # Convert the dataset to a NumPy array.
        data = np.array(dataset)

        # Compute maximum amplitudes for each channel.
        max_e = np.max(data[:, 0])
        max_n = np.max(data[:, 1])
        max_z = np.max(data[:, 2])

        # Append the computed values to the corresponding lists.
        max_amplitudes_e.append(max_e)
        max_amplitudes_n.append(max_n)
        max_amplitudes_z.append(max_z)
        #source_magnitudes.append(dataset.attrs['source_magnitude'])

        if(plotting):
            # %% Plot waveforms of the current event
            fig = plt.figure(figsize=(8, 12), dpi=300)

            # Subplot 1: E channel
            ax1 = fig.add_subplot(411)
            ax1.plot(data[:, 0], 'k')
            ymin, ymax = ax1.get_ylim()
            ax1.vlines(dataset.attrs['p_arrival_sample'], ymin, ymax, color='b', linewidth=2, label='P-arrival')
            ax1.vlines(dataset.attrs['s_arrival_sample'], ymin, ymax, color='r', linewidth=2, label='S-arrival')
            ax1.vlines(dataset.attrs['coda_end_sample'], ymin, ymax, color='aqua', linewidth=2, label='Coda End')
            ax1.legend(loc='upper right', prop={'weight': 'bold'})
            ax1.set_xticklabels([])  # Hide x-axis tick labels
            ax1.set_ylabel('Amplitude counts', fontsize=12)
            # print(f"E Channel - max: {max_e} | min: {np.min(data[:, 0])} | magnitude: {dataset.attrs['source_magnitude']}")

            # Subplot 2: N channel
            ax2 = fig.add_subplot(412)
            ax2.plot(data[:, 1], 'k')
            ymin, ymax = ax2.get_ylim()
            ax2.vlines(dataset.attrs['p_arrival_sample'], ymin, ymax, color='b', linewidth=2, label='P-arrival')
            ax2.vlines(dataset.attrs['s_arrival_sample'], ymin, ymax, color='r', linewidth=2, label='S-arrival')
            ax2.vlines(dataset.attrs['coda_end_sample'], ymin, ymax, color='aqua', linewidth=2, label='Coda End')
            ax2.legend(loc='upper right', prop={'weight': 'bold'})
            ax2.set_xticklabels([])
            ax2.set_ylabel('Amplitude counts', fontsize=12)
            # print(f"N Channel - max: {max_n} | min: {np.min(data[:, 1])} | magnitude: {dataset.attrs['source_magnitude']}")

            # Subplot 3: Z channel
            ax3 = fig.add_subplot(413)
            ax3.plot(data[:, 2], 'k')
            ymin, ymax = ax3.get_ylim()
            ax3.vlines(dataset.attrs['p_arrival_sample'], ymin, ymax, color='b', linewidth=2, label='P-arrival')
            ax3.vlines(dataset.attrs['s_arrival_sample'], ymin, ymax, color='r', linewidth=2, label='S-arrival')
            ax3.vlines(dataset.attrs['coda_end_sample'], ymin, ymax, color='aqua', linewidth=2, label='Coda End')
            ax3.legend(loc='upper right', prop={'weight': 'bold'})
            ax3.set_xticklabels([])
            ax3.set_ylabel('Amplitude counts', fontsize=12)
            # print(f"Z Channel - max: {max_z} | min: {np.min(data[:, 2])} | magnitude: {dataset.attrs['source_magnitude']}")

            # Subplot 4: Combined channels (E in blue, N in green, Z in red)
            ax4 = fig.add_subplot(414)
            ax4.plot(data[:, 0], color='blue', label='E')
            ax4.plot(data[:, 1], color='green', label='N')
            ax4.plot(data[:, 2], color='red', label='Z')
            ax4.legend(loc='upper right', prop={'weight': 'bold'})
            ax4.set_ylabel('Amplitude counts', fontsize=12)
            ax4.set_xlabel('Time (samples)')
            plt.tight_layout()
            plt.show()

            # Optionally, print all dataset attributes for reference.
            for at in dataset.attrs:
                print(at, dataset.attrs[at])

        # inp = input("Press a key to plot the next waveform (enter 'r' to restart current chunk if needed): ")
        # if inp == "r":
        #     continue

    # Close the HDF5 file after processing the current chunk.
    dtfl.close()

# %% Convert the lists to NumPy arrays for easier manipulation.
#source_magnitudes = np.array(source_magnitudes, dtype=float)
source_magnitudes = np.linspace(0, 0, num=len(max_amplitudes_e))
max_amplitudes_e = np.array(max_amplitudes_e, dtype=float)
max_amplitudes_n = np.array(max_amplitudes_n, dtype=float)
max_amplitudes_z = np.array(max_amplitudes_z, dtype=float)

# %% Create separate scatter plots for each channel.
print("Minimum magnitude:", np.min(source_magnitudes))
print("Maximum magnitude:", np.max(source_magnitudes))

# E Channel Scatter Plot:
plt.figure(figsize=(10, 6))
plt.scatter(source_magnitudes, max_amplitudes_e, alpha=0.7)
plt.xlabel("Source Magnitude")
plt.ylabel("Maximum Amplitude (E counts)")
plt.title("Scatter Plot: Source Magnitude vs. E Channel Maximum Amplitude")
plt.tight_layout()
plt.show()

# N Channel Scatter Plot:
plt.figure(figsize=(10, 6))
plt.scatter(source_magnitudes, max_amplitudes_n, alpha=0.7)
plt.xlabel("Source Magnitude")
plt.ylabel("Maximum Amplitude (N counts)")
plt.title("Scatter Plot: Source Magnitude vs. N Channel Maximum Amplitude")
plt.tight_layout()
plt.show()

# Z Channel Scatter Plot:
plt.figure(figsize=(10, 6))
plt.scatter(source_magnitudes, max_amplitudes_z, alpha=0.7)
plt.xlabel("Source Magnitude")
plt.ylabel("Maximum Amplitude (Z counts)")
plt.title("Scatter Plot: Source Magnitude vs. Z Channel Maximum Amplitude")
plt.tight_layout()
plt.show()
