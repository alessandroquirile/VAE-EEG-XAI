from mne.io import read_raw_bdf


def read_bdf(path, subject):
    eog = ['EXG1', 'EXG2', 'EXG3', 'EXG4']
    misc = ['EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']
    raw = read_raw_bdf(path + subject, eog=eog, misc=misc, stim_channel='status', preload=True, verbose=False)
    return raw


def get_indices_where_video_start(raw):
    # The status channel contains markers sent from the stimuli presentation PC,
    # indicating when trials start and end.
    status_channel = raw.get_data()[47]
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


def crop(raw, id_trial21):
    sample_rate = get_sample_rate(raw)
    min1 = sample_rate * 60
    tmin = id_trial21 / sample_rate
    tmax = (id_trial21 + min1) / sample_rate
    raw_trial21 = raw.crop(tmin=tmin, tmax=tmax)
    return raw_trial21


def get_sample_rate(raw):
    return int(raw.info['sfreq'])


def save(plt, file_name):
    fig = plt.gcf()
    fig.canvas.draw()
    fig.savefig(file_name)
