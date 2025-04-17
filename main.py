import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, iirnotch, welch
from scipy.integrate import simpson
import csv
import json
import os

# Filters
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter(order, [(2*lowcut)/fs, (2*highcut)/fs], btype='band')
    return lfilter(b, a, data)

def notch_filter(data, fn, fs, q):
    b, a = iirnotch(fn, q, fs)
    return lfilter(b, a, data)

def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simpson(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simpson(psd, dx=freq_res)
    return bp

def extract_user_features(n):
    # File names
    parameters_infile_name = f"source/sub-{n if n>10 else f"0{n}"}/sub-{n if n>10 else f"0{n}"}_task-ImaginedEmotion_eeg.json"
    anotations_infile_name = f"source/sub-{n if n>10 else f"0{n}"}/sub-{n if n>10 else f"0{n}"}_task-ImaginedEmotion_events.txt"
    data_infile_name = f"source/sub-{n if n>10 else f"0{n}"}/sub-{n if n>10 else f"0{n}"}_task-ImaginedEmotion_eeg.set"
    features_outfile_name = f"result/sub-{n if n>10 else f"0{n}"}/sub-{n if n>10 else f"0{n}"}_task-ImaginedEmotion_features.csv"
    feature_dimensions_outfile_name = f"result/sub-{n if n>10 else f"0{n}"}/sub-{n if n>10 else f"0{n}"}_task-ImaginedEmotion_feature_dimensions.csv"
    os.makedirs(os.path.dirname(features_outfile_name), exist_ok=True)
    os.makedirs(os.path.dirname(feature_dimensions_outfile_name), exist_ok=True)

    # Constants
    fl = 1    # Lower frequency of band pass filer (Hz)
    fh = 80   # Higher frequency of band pass filter (Hz)
    q = 100   # Quality factor of the notch filter
    baseline_name = "relax"
    baseline_end_name = "ImaginationSuggestions"
    emotion_names = ["awe", "frustration", "joy", "anger", "happy", "sad", "love", "grief", "fear", "compassion", "jealousy", "content", "grief", "relief", "excite", "disgust"]
    emotion_end_name = "exit"
    freq_band_names = ["delta", "theta", "alpha", "beta", "gamma"]  # names of the brain waves
    freq_bands = np.asanyarray([[1, 4], [4, 8], [8, 12], [12 ,30], [30, 80]])  # frequency ranges of the brain waves (Hz)
    no_freq_bands = len(freq_band_names)
    window_length = 4  # length of Welch's window (sec)

    # Reading parameters
    fn = 0
    no_channels = 0
    T = 0
    fs = 0
    with open(parameters_infile_name, mode="r", encoding="utf-8") as constants_infile:
        constants_json = json.load(constants_infile)
        fn = float(constants_json["PowerLineFrequency"])
        no_channels = int(constants_json["EEGChannelCount"])
        T = int(constants_json["RecordingDuration"])
        fs = int(constants_json["SamplingFrequency"])

    # Reading baseline annotations
    T_max = 0
    baseline_times = np.zeros(2)
    with open(anotations_infile_name, 'r') as text_file:
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
    with open(anotations_infile_name, 'r') as text_file:
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
    no_emotions = len(emotion_times)  # to account for the baseline

    # Reading EEG data
    raw = mne.io.read_raw_eeglab(data_infile_name)
    data, times = raw.get_data(return_times=True, units='uV')

    # Filtering data  # TODO: consolidate into one matrix
    bp_data = np.zeros((no_channels, T*fs))
    filtered_data = np.zeros((no_channels, T*fs))
    for channel in range(no_channels):
        bp_data[channel] = butter_bandpass_filter(data[channel], fl, fh, fs, order=8)
        filtered_data[channel] = notch_filter(bp_data[channel], fn, fs, q)

    # Epoching data
    epoched_data = np.zeros((no_channels, no_emotions+1, int(T_max*fs)))
    # baseline epoching
    baseline_start_index, = np.where(np.isclose(times, baseline_times[0]))
    baseline_end_index, = np.where(np.isclose(times, baseline_times[1]))
    baseline_start_index = baseline_start_index[0]
    baseline_end_index = baseline_end_index[0]
    for channel_no in range(no_channels):
        epoched_data[channel_no][0][0:(baseline_end_index-baseline_start_index)] = data[channel_no][baseline_start_index:baseline_end_index]
    # emotions epoching
    for emotion_no in range(no_emotions):
        start_index, = np.where(np.isclose(times, emotion_times[emotion_no][1]))
        end_index, = np.where(np.isclose(times, emotion_times[emotion_no][2]))
        start_index = start_index[0]
        end_index = end_index[0]
        for channel_no in range(no_channels):
            epoched_data[channel_no][emotion_no+1][0:(end_index - start_index)] = data[channel_no][start_index:end_index]

    # Feature extraction
    baseline_bp = np.zeros((no_channels, no_freq_bands))
    emotion_bp_db = np.zeros((no_channels, no_emotions, no_freq_bands))
    # baseline exraction
    for channel_no in range(no_channels):
        for freq_band_no in range(no_freq_bands):
            baseline_bp[channel_no][freq_band_no] = bandpower(epoched_data[channel_no][0], fs, freq_bands[freq_band_no])
    # emotion extraction
    for channel_no in range(no_channels):
        for emotion_no in range(no_emotions):
            for freq_band_no in range(no_freq_bands):
                temp_bp = bandpower(epoched_data[channel_no][emotion_no+1], fs, freq_bands[freq_band_no], window_length, True)/baseline_bp[channel_no][freq_band_no]
                emotion_bp_db[channel_no][emotion_no][freq_band_no] = 10*np.log10(temp_bp) if temp_bp>0 else -1

    with open(features_outfile_name, "w") as features_outfile:
        emotion_bp_db.flatten().tofile(features_outfile, sep=",")
        features_outfile.flush()
    with open(feature_dimensions_outfile_name, "w") as feature_dimensions_outfile:
        np.asanyarray(emotion_bp_db.shape).tofile(feature_dimensions_outfile, sep=",")
        feature_dimensions_outfile.flush()

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

def main():
    extract_user_features(2)

if __name__ == "__main__":
    main()
