import numpy as np
import mne
import matplotlib.pyplot as plt

file_name = "sub-02_task-ImaginedEmotion_eeg.set"
raw = mne.io.read_raw_eeglab(f"./sources/{file_name}")
data, times = raw.get_data(return_times=True)

times = times
data = data[0]
plt.plot(times, data)
plt.show()
