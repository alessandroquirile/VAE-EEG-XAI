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
    # The status channel contains markers sent from the stimuli presentation PC,
    # indicating when trials start and end.
    STATUS_CHANNEL_ID = 47
    status_channel = raw.get_data()[STATUS_CHANNEL_ID]
    min1 = get_sample_rate(raw) * 60
    idx_starts = []
    i = 0
    while i <= (status_channel.shape[0] - min1):
        if status_channel[i] == 4:  # 4: Start of music video playback
            idx_starts.append(i)  # salva gli indici di quando inziano la visione dei video
            i = i + min1  # avanti di un minuto
        else:
            i += 1
    return idx_starts


def crop(raw: mne.io.Raw, index_trial: int) -> mne.io.Raw:
    sample_rate = get_sample_rate(raw)
    min1 = sample_rate * 60
    tmin = index_trial / sample_rate
    tmax = (index_trial + min1) / sample_rate
    cropped_raw = raw.crop(tmin=tmin, tmax=tmax)
    return cropped_raw


def get_sample_rate(raw: mne.io.Raw) -> int:
    return int(raw.info['sfreq'])


def save(plt, file_name: str):
    fig = plt.gcf()
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


def create_filename(folder, subject, trial_count, extension):
    subject_no_extension = subject.split(".")[0]
    two_digits_trial_count = str(trial_count).zfill(2)
    filename = f"{folder}{subject_no_extension}_trial{two_digits_trial_count}{extension}"
    return filename


def save_plot(folder: str, subject, trial_count, data, info):
    raw_filtered = mne.io.RawArray(data, info)
    plot_raw(raw_filtered, duration=60, scalings=20e-5)
    # plt.show()
    filename = create_filename(folder, subject, trial_count, '.png')
    os.makedirs(folder, exist_ok=True)
    save(plt, filename)
    plt.close()


def save_events(folder, subject, trial_count, events, evt_in_sec):
    filename = create_filename(folder, subject, trial_count, '.txt')
    os.makedirs(folder, exist_ok=True)
    with open(filename, 'w') as f:
        f.write('Events detected: {}\n'.format(len(evt_in_sec)))
        f.write('Events: {}\n'.format(events))
        f.write('Events in seconds: {}\n'.format(evt_in_sec))


def calculate_threshold(filtered_data):
    return (np.max(filtered_data) - np.min(filtered_data)) / 2


def convert_in_seconds(events, index, raw):
    events_indices = events[:, 0]
    return (events_indices - index) / get_sample_rate(raw)
