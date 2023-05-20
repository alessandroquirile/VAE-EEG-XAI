import os

import mne.io
import numpy as np
from matplotlib import pyplot as plt
from mne.io import read_raw_bdf
from mne.viz import plot_raw


def read_bdf(path: str, subject: str) -> mne.io.Raw:
    eog = ['EXG1', 'EXG2', 'EXG3', 'EXG4']
    misc = ['EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']
    raw = read_raw_bdf(path + subject, eog=eog, misc=misc, stim_channel='status', preload=True, verbose=False)
    return raw


def get_indices_where_video_start(raw: mne.io.Raw) -> list[int]:
    STATUS_CHANNEL_ID = 47
    sample_rate = get_sample_rate(raw)
    n_samples_in_1_minute = _get_number_of_samples_in_1_minute(sample_rate)
    status_channel = raw.get_data()[STATUS_CHANNEL_ID]
    idx_starts = []
    i = 0
    while i <= (status_channel.shape[0] - n_samples_in_1_minute):
        if status_channel[i] == 4:
            idx_starts.append(i)
            i += n_samples_in_1_minute
        else:
            i += 1
    return idx_starts


def crop(raw: mne.io.Raw, sample_index: int) -> mne.io.Raw:
    sample_rate = get_sample_rate(raw)
    n_samples_in_1_minute = _get_number_of_samples_in_1_minute(sample_rate)
    tmin = sample_index / sample_rate
    tmax = (sample_index + n_samples_in_1_minute) / sample_rate
    cropped_raw = raw.crop(tmin=tmin, tmax=tmax)
    return cropped_raw


def get_sample_rate(raw: mne.io.Raw) -> int:
    return int(raw.info['sfreq'])


def save(plot, file_name: str):
    fig = plot.gcf()
    fig.canvas.draw()
    fig.savefig(file_name)


def get_subjects(path: str) -> list[str]:
    subjects = []
    if os.path.exists(path):
        # s01-s22 are the only who have videos
        for i in range(1, 23):
            subject = f's{i:02d}.bdf'
            subject_path = os.path.join(path, subject)
            if os.path.exists(subject_path):
                subjects.append(subject)
    return subjects


def calculate_threshold(filtered_data):
    return (np.max(filtered_data) - np.min(filtered_data)) / 2


def save_plot(folder_name: str, subject: str, trial: int, data, info):
    raw_filtered = mne.io.RawArray(data, info)
    plot_raw(raw_filtered, duration=60, scalings=20e-5)
    # plt.show()
    filename = _create_filename(folder_name, subject, trial, 'png')
    os.makedirs(folder_name, exist_ok=True)
    save(plt, filename)
    plt.close()


def save_events(folder_name: str, subject: str, trial: int, events, sample_index, sample_rate):
    filename = _create_filename(folder_name, subject, trial, 'txt')
    os.makedirs(folder_name, exist_ok=True)
    events_in_seconds = _convert_in_seconds(events, sample_index, sample_rate)
    with open(filename, 'w') as f:
        f.write('Events detected: {}\n'.format(len(events_in_seconds)))
        f.write('Events: {}\n'.format(events))
        f.write('Events in seconds: {}\n'.format(events_in_seconds))


def _create_filename(folder_name: str, subject: str, trial: int, extension: str):
    subject_no_extension = subject.split(".")[0]
    two_digits_trial_count = str(trial).zfill(2)
    filename = f"{folder_name}/{subject_no_extension}_trial{two_digits_trial_count}.{extension}"
    return filename


def _get_number_of_samples_in_1_minute(sample_rate: int):
    SECONDS_IN_1_MINUTE = 60
    n_samples_in_1_minute = sample_rate * SECONDS_IN_1_MINUTE
    return n_samples_in_1_minute


def _convert_in_seconds(events, sample_index, sample_rate: int):
    events_indices = events[:, 0]
    return (events_indices - sample_index) / sample_rate
