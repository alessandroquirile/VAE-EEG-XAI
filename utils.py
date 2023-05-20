import mne.io
from mne.io import read_raw_bdf


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
