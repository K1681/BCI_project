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
    T = 5492  # Recording time (seconds) of each user.
    fs = 256  # Rampling rate (Hz)
    fl = 1    # Lower frequency of band pass filer
    fh = 80   # Higher frequency of band pass filter
    fn = 60   # Line freequency, to be removed by notch filter
    q = 100   # Quality factor of the notch filter
    no_channels = 214  # Number of channels
    baseline_name = "relax"
    baseline_end_name = "ImaginationSuggestions"
    emotion_names = ["awe", "frustration", "joy", "anger", "happy", "sad", "love", "grief", "fear", "compassion", "jealousy", "content", "grief", "relief", "excite", "disgust"]
    emotion_end_name = "exit"
    emotions_file_name = "sub-02_task-ImaginedEmotion_events.txt"
    data_file_name = "sub-02_task-ImaginedEmotion_eeg.set"

    # Reading baseline annotations
    # N = len(emotion_names)  # Number of emotions to detect
    T_max = 0
    baseline_times = np.zeros(2)
    with open(f"sources/{emotions_file_name}", 'r') as text_file:
        tsv_values = list(csv.reader(text_file, delimiter="\t"))
        for row in range(len(tsv_values)):
            if(tsv_values[row][6].strip() == baseline_name):
                baseline_times[0] = float(tsv_values[row][0].strip())
                while(tsv_values[row][6].strip() != baseline_end_name):
                    row += 1
                baseline_times[1] = float(tsv_values[row][0].strip())
    T_max = baseline_times[1] - baseline_times[0]

    # Reading emotion annotations
    emotion_times = []
    with open(f"sources/{emotions_file_name}", 'r') as text_file:
        tsv_values = list(csv.reader(text_file, delimiter="\t"))
        counter = 0
        for row in range(len(tsv_values)):
            if(tsv_values[row][6].strip() in emotion_names):
                emotion_times.append([0, 0, 0])
                emotion_times[counter][0] = emotion_names.index(tsv_values[row][6].strip())
                emotion_times[counter][1] = float(tsv_values[row][0].strip())
                while(tsv_values[row][6].strip() != emotion_end_name):
                    row += 1
                emotion_times[counter][2] = float(tsv_values[row][0].strip())
                if((emotion_times[counter][2] - emotion_times[counter][1]) > T_max):
                    T_max = emotion_times[counter][2] - emotion_times[counter][1]
                counter += 1
    emotion_times = np.asarray(emotion_times, dtype=np.float32)

    # Reading EEG data
    raw = mne.io.read_raw_eeglab(f"./sources/{data_file_name}")
    data, times = raw.get_data(return_times=True, units='uV')

    # Filtering data  # TODO: consolidate into one matrix
    bp_data = np.zeros((no_channels, T*fs))
    filtered_data = np.zeros((no_channels, T*fs))
    for channel in range(no_channels):
        bp_data[channel] = butter_bandpass_filter(data[channel], fl, fh, fs, order=8)
        filtered_data[channel] = notch_filter(bp_data[channel], fn, fs, q)

    # Epoching data
    epoched_data = np.zeros((no_channels, len(emotion_times), int(T_max*fs)))
    # epoched_data = [[] for _ in range(no_channels)]
    for emotion_no in range(len(emotion_times)):
        start_index, = np.where(np.isclose(times, emotion_times[emotion_no][1]))
        end_index, = np.where(np.isclose(times, emotion_times[emotion_no][2]))
        start_index = start_index[0]
        end_index = end_index[0]
        for channel_no in range(no_channels):
            epoched_data[channel_no][emotion_no][0:(end_index - start_index)] = data[channel_no][start_index:end_index]
    epoched_data = np.asarray(epoched_data, dtype=np.float32)


    """
    # Visualization
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
