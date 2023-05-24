import os

import mne
from matplotlib import pyplot as plt
from mne.filter import filter_data
from mne.viz import plot_raw

from utils import get_sample_rate

def save_plot(folder_name: str, l_freq, h_freq, raw, subject, trial):
    data = raw.get_data()
    sample_rate = get_sample_rate(raw)
    filtered_data = filter_data(data, sample_rate, l_freq, h_freq)
    _save_plot(folder_name, subject, trial, filtered_data, raw.info)


def save_events(folder_name: str, subject: str, trial: int, events, sample_index, sample_rate):
    filename = _create_filename(folder_name, subject, trial, 'txt')
    os.makedirs(folder_name, exist_ok=True)
    events_in_seconds = _convert_in_seconds(events, sample_index, sample_rate)
    with open(filename, 'w') as f:
        f.write('Events detected: {}\n'.format(len(events_in_seconds)))
        f.write('Events: {}\n'.format(events))
        f.write('Events in seconds: {}\n'.format(events_in_seconds))


def _save_plot(folder_name: str, subject: str, trial: int, data, info):
    raw_filtered = mne.io.RawArray(data, info)
    plot_raw(raw_filtered, duration=60, scalings=20e-5)
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
