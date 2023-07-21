import os

import mne
from matplotlib import pyplot as plt
from mne.filter import filter_data
from mne.viz import plot_raw

from utils import get_sample_rate


def save_plot(folder_name: str, l_freq, h_freq, raw, subject, trial):
    print(f"\nSaving {subject}_trial{trial} plot to {folder_name} folder")
    data = raw.get_data()
    sample_rate = get_sample_rate(raw)
    filtered_data = filter_data(data, sample_rate, l_freq, h_freq, verbose=False)
    _save_plot(folder_name, subject, trial, filtered_data, raw.info)


def save_events(folder_name: str, subject: str, trial: int, events, sample_index, sample_rate,
                lowest_peak, highest_peak, magic_number,
                trial_lowest_peak, trial_highest_peak, thresh):
    # print(f"\nSaving {subject}_trial{trial} events to {folder_name} folder")
    filename = _create_filename(folder_name, subject, trial, 'txt')
    os.makedirs(folder_name, exist_ok=True)
    events_in_seconds = _convert_in_seconds(events, sample_index, sample_rate)
    with open(filename, 'w') as f:
        f.write('Events detected: {}\n'.format(len(events_in_seconds)))
        f.write('Events: {}\n'.format(events))
        f.write('Events in seconds: {}\n'.format(events_in_seconds))
        f.write('k: {}\n'.format(magic_number))
        f.write('threshold: {:.8f}\n'.format(thresh))
        f.write('Subject lowest peak: {:.8f}\n'.format(lowest_peak))
        f.write('Subject highest peak: {:.8f}\n'.format(highest_peak))
        f.write('Subject peaks difference: {:.8f}\n'.format(highest_peak - lowest_peak))
        f.write('Trial lowest peak: {:.8f}\n'.format(trial_lowest_peak))
        f.write('Trial highest peak: {:.8f}\n'.format(trial_highest_peak))
        f.write('Trial peaks difference: {:.8f}\n'.format(trial_highest_peak - trial_lowest_peak))


def _save_plot(folder_name: str, subject: str, trial: int, data, info):
    raw_filtered = mne.io.RawArray(data, info)
    plot_raw(raw_filtered, duration=60, scalings=20e-5, show=False)
    # plt.show()
    filename = _create_filename(folder_name, subject, trial, 'png')
    os.makedirs(folder_name, exist_ok=True)
    _save(plt, filename)
    plt.close()


def _create_filename(folder_name: str, subject: str, trial: int, extension: str):
    subject_no_extension = subject.split(".")[0]
    two_digits_trial_count = str(trial).zfill(2)
    filename = f"{folder_name}/{subject_no_extension}_trial{two_digits_trial_count}.{extension}"
    return filename


def _save(plot, file_name: str):
    fig = plot.gcf()
    fig.canvas.draw()
    fig.savefig(file_name)


def _convert_in_seconds(events, sample_index, sample_rate: int):
    events_indices = events[:, 0]
    return (events_indices - sample_index) / sample_rate
