import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, iirnotch
import csv

# Filters
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter(order, [(2*lowcut)/fs, (2*highcut)/fs], btype='band')
    return lfilter(b, a, data)

def notch_filter(data, fn, fs, q):
    b, a = iirnotch(fn, q, fs)
    return lfilter(b, a, data)

def main():
    # Constants
    T = 5492
    fs = 256
    fl = 1
    fh = 80
    fn = 60
    q = 100
    no_channels = 214
    feeling_names = ["awe", "frustration", "joy", "anger", "happy", "sad", "love", "grief", "fear", "compassion", "jealousy", "content", "grief", "relief", "excite", "disgust", "relax"]
    N = len(feeling_names)

    # Reading EEG data
    data_file_name = "sub-02_task-ImaginedEmotion_eeg.set"
    raw = mne.io.read_raw_eeglab(f"./sources/{data_file_name}")
    data, times = raw.get_data(return_times=True, units='uV')

    # Reading annotations
    T_max = 0
    feeling_times = np.zeros((N, 3))
    feelings_file_name = "sub-02_task-ImaginedEmotion_events.txt"
    with open(f"sources/{feelings_file_name}", 'r') as text_file:
        tsv_values = list(csv.reader(text_file, delimiter="\t"))
        counter = 0
        for row in range(len(tsv_values)):
            if(tsv_values[row][6].strip() in feeling_names):
                feeling_times[counter][0] = feeling_names.index(tsv_values[row][6].strip())
                feeling_times[counter][1] = float(tsv_values[row][0].strip())
                while(tsv_values[row][6].strip() != "exit"):
                    row += 1
                feeling_times[counter][2] = float(tsv_values[row][0].strip())
                counter += 1
                if((feeling_times[counter][2] - feeling_times[counter][1]) > T_max):
                    T_max = feeling_times[counter][2] - feeling_times[counter][1]

    # Filtering data  # TODO: consolidate into one matrix
    bp_data = np.zeros((no_channels, T*fs))
    filtered_data = np.zeros((no_channels, T*fs))
    for channel in range(no_channels):
        bp_data[channel] = butter_bandpass_filter(data[channel], fl, fh, fs, order=8)
        filtered_data[channel] = notch_filter(bp_data[channel], fn, fs, q)

    # Epoching data
    epoched_data = np.zeros((no_channels, N, T_max*fs))
    for channel in range(no_channels):
        for feeling in range(N):
            start, end = feeling_times[feeling_times[:].index(feeling)][1:2]  # TODO: fix searching.
            epoched_data[channel][feeling] = filtered_data[channel][start : end ]

    print(epoched_data.shape)

    """
    x_right = 10
    # Raw data
    plt.subplot(3, 1, 1)
    plt.plot(times, data[0])
    plt.xlim(0, x_right)

    # Band Pass filter
    plt.subplot(3, 1, 2)
    plt.plot(times, bp_data)
    plt.xlim(0, x_right)

    # Notch filter
    plt.subplot(3, 1, 3)
    plt.plot(times, np_data)
    plt.xlim(0, x_right)

    plt.show()
    """

if __name__ == "__main__":
    main()
