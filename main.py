import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, iirnotch

# Filters
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter(order, [(2*lowcut)/fs, (2*highcut)/fs], btype='band')
    return lfilter(b, a, data)

def notch_filter(data, fn, fs, q):
    b, a = iirnotch(fn, q, fs)
    return lfilter(b, a, data)

def main():
    # Reading file
    file_name = "sub-02_task-ImaginedEmotion_eeg.set"
    raw = mne.io.read_raw_eeglab(f"./sources/{file_name}")
    data, times = raw.get_data(return_times=True, units='uV')

    # Filtering data
    fs = 256
    fl = 1
    fh = 80
    fn = 60
    q = 100
    bp_data = butter_bandpass_filter(data[0], fl, fh, fs, order=8)
    np_data = notch_filter(bp_data, fn, fs, q)

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
