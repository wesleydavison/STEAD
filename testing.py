#%%
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt

file_name = r"C:\Users\cadas\Box\ECSTATIC - General\02 datasets (public)\STanford EArthquake Dataset (STEAD)\chunk2.hdf5"
csv_file = r"C:\Users\cadas\Box\ECSTATIC - General\02 datasets (public)\STanford EArthquake Dataset (STEAD)\chunk2.csv"

chunksize = 10000

# reading the csv file into a dataframe:
chunks = pd.read_csv(csv_file, chunksize=chunksize)
for chunk in chunks:
    print(f'total events in csv file: {len(chunk)}')
    # filterering the dataframe
    chunk = chunk[(chunk.trace_category == 'earthquake_local') & (chunk.source_distance_km <= 20) & (chunk.source_magnitude > 3)]
    print(f'total events selected: {len(chunk)}')

    # making a list of trace names for the selected data
    ev_list = chunk['trace_name'].to_list()

    # retrieving selected waveforms from the hdf5 file:
    dtfl = h5py.File(file_name, 'r')
    for c, evi in enumerate(ev_list):
        dataset = dtfl.get('data/'+str(evi))
        # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
        data = np.array(dataset)

        # Create the figure
        fig = plt.figure(figsize=(8, 12), dpi=300)  # Adjusted figsize for 4 subplots

        # Subplot 1: E channel
        ax1 = fig.add_subplot(411)
        ax1.plot(data[:, 0], 'k')
        ymin, ymax = ax1.get_ylim()
        ax1.vlines(dataset.attrs['p_arrival_sample'], ymin, ymax, color='b', linewidth=2, label='P-arrival')
        ax1.vlines(dataset.attrs['s_arrival_sample'], ymin, ymax, color='r', linewidth=2, label='S-arrival')
        ax1.vlines(dataset.attrs['coda_end_sample'], ymin, ymax, color='aqua', linewidth=2, label='Coda End')
        ax1.legend(loc='upper right', prop={'weight': 'bold'})
        ax1.set_xticklabels([])  # Hide x tick labels
        ax1.set_ylabel('Amplitude counts', fontsize=12)
        print(
            f"E Channel - max:{np.max(data[:, 0])} - min:{np.min(data[:, 0])} - magnitude:{dataset.attrs['source_magnitude']}")

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
        print(
            f"N Channel - max:{np.max(data[:, 1])} - min:{np.min(data[:, 1])} - magnitude:{dataset.attrs['source_magnitude']}")

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
        print(
            f"Z Channel - max:{np.max(data[:, 2])} - min:{np.min(data[:, 2])} - magnitude:{dataset.attrs['source_magnitude']}")

        # Subplot 4: Combined mix of E, N, and Z channels, each in a different color
        ax4 = fig.add_subplot(414)
        ax4.plot(data[:, 0], color='blue', label='E')  # E channel in blue
        ax4.plot(data[:, 1], color='green', label='N')  # N channel in green
        ax4.plot(data[:, 2], color='red', label='Z')  # Z channel in red
        ymin, ymax = ax4.get_ylim()
        # (If you wish to add vertical lines, uncomment and modify as needed)
        # ax4.vlines(dataset.attrs['p_arrival_sample'], ymin, ymax, color='b', linewidth=2, label='P-arrival')
        # ax4.vlines(dataset.attrs['s_arrival_sample'], ymin, ymax, color='r', linewidth=2, label='S-arrival')
        # ax4.vlines(dataset.attrs['coda_end_sample'], ymin, ymax, color='aqua', linewidth=2, label='Coda End')
        ax4.legend(loc='upper right', prop={'weight': 'bold'})
        ax4.set_ylabel('Amplitude counts', fontsize=12)
        ax4.set_xlabel('Time (samples)')
        plt.tight_layout()
        plt.show()

        # Print dataset attributes if needed
        for at in dataset.attrs:
            print(at, dataset.attrs[at])

        inp = input("Press a key to plot the next waveform!")
        if inp == "r":
            continue
#%%
