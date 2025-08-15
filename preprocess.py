import sys
sys.path.append('D:\\anaconda\\lib\\site-packages')
import matplotlib.pyplot as plt
from scipy.signal import decimate, butter, filtfilt
import numpy as np

def downsample(signal, original_rate, target_rate):
    factor = original_rate // target_rate
    downsampled_signal = decimate(signal, factor, zero_phase=True)
    return downsampled_signal

def butter_filter(cutoff, fs, btype='low', order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a

def filter_signal(data, cutoff, fs, btype='low', order=4):
    b, a = butter_filter(cutoff, fs, btype, order=order)
    y = filtfilt(b, a, data)
    return y


def moving_average(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')


# Main function
def process_ppg_signal(ppg_signal, original_rate=400, target_rate=125):
    plot_signal(ppg_signal, 'Raw Signal')
    # Step 1: Downsample the signal
    downsampled_signal = downsample(ppg_signal, original_rate, target_rate)
    # plot_signal(downsampled_signal, 'Downsampled Signal')

    # Step 2: Highpass Filtering (0.05 Hz)
    highpassed_signal = filter_signal(downsampled_signal, cutoff=0.5, fs=target_rate, btype='high')
    # plot_signal(highpassed_signal, 'Highpass Filtered Signal (0.5 Hz)')

    # Step 3: Lowpass Filtering (10 Hz)
    lowpassed_signal = filter_signal(highpassed_signal, cutoff=5, fs=target_rate, btype='low')
    plot_signal(lowpassed_signal[2000:4000], 'Cleaned_signal')
    plot_signal(lowpassed_signal[2500:3000], 'Small Size')

    # Step 4: Smoothing Filter
    # smoothed_signal = moving_average(lowpassed_signal, window_size=5)
    # plot_signal(smoothed_signal, 'Smoothed Signal')

    return lowpassed_signal

def plot_signal(signal, title):
    plt.figure(figsize=(12, 6))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.show()
# Example usage
if __name__ == "__main__":
    data_array = np.loadtxt('ZZdata1.txt')

    cleaned_signal = process_ppg_signal(data_array)

    # Output the cleaned signal for inspection
    # print(cleaned_signal)