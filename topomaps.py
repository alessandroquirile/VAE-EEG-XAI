import matplotlib.pyplot as plt
from mne.preprocessing import find_eog_events

from eeg_constants import *
from savers import save_plot, save_events
from utils import *


# monkey patching, per non creare naso e orecchie
def _make_head_outlines_new(sphere, pos, outlines, clip_origin):
    """Check or create outlines for topoplot."""
    assert isinstance(sphere, np.ndarray)
    x, y, _, radius = sphere
    del sphere

    if outlines in ("head", None):
        ll = np.linspace(0, 2 * np.pi, 101)
        head_x = np.cos(ll) * radius + x
        head_y = np.sin(ll) * radius + y
        dx = np.exp(np.arccos(np.deg2rad(12)) * 1j)
        dx, dy = dx.real, dx.imag
        # nose_x = np.array([-dx, 0, dx]) * radius + x
        # nose_y = np.array([dy, 1.15, dy]) * radius + y
        # ear_x = np.array(
        #     [0.497, 0.510, 0.518, 0.5299, 0.5419, 0.54, 0.547, 0.532, 0.510, 0.489]
        # ) * (radius * 2)
        # ear_y = (
        #     np.array(
        #         [
        #             0.0555,
        #             0.0775,
        #             0.0783,
        #             0.0746,
        #             0.0555,
        #             -0.0055,
        #             -0.0932,
        #             -0.1313,
        #             -0.1384,
        #             -0.1199,
        #         ]
        #     )
        #     * (radius * 2)
        #     + y
        # )

        if outlines is not None:
            # Define the outline of the head, ears and nose
            outlines_dict = dict(
                head=(head_x, head_y),
                # nose=(nose_x, nose_y),
                # ear_left=(-ear_x + x, ear_y),
                # ear_right=(ear_x + x, ear_y),
            )
        else:
            outlines_dict = dict()

        # Make the figure encompass slightly more than all points
        # We probably want to ensure it always contains our most
        # extremely positioned channels, so we do:
        mask_scale = max(1.0, np.linalg.norm(pos, axis=1).max() * 1.01 / radius)
        outlines_dict["mask_pos"] = (mask_scale * head_x, mask_scale * head_y)
        clip_radius = radius * mask_scale
        outlines_dict["clip_radius"] = (clip_radius,) * 2
        outlines_dict["clip_origin"] = clip_origin
        outlines = outlines_dict

    elif isinstance(outlines, dict):
        if "mask_pos" not in outlines:
            raise ValueError("You must specify the coordinates of the image " "mask.")
    else:
        raise ValueError("Invalid value for `outlines`.")

    return outlines


# monkey patching, per non far disegnare l'outline
def _draw_outlines_new(ax, outlines):
    """Draw the outlines for a topomap."""
    from matplotlib import rcParams

    outlines_ = {k: v for k, v in outlines.items() if k not in ["patch"]}
    for key, (x_coord, y_coord) in outlines_.items():
        if "mask" in key or key in ("clip_radius", "clip_origin"):
            continue
        ax.plot(
            x_coord,
            y_coord,
            color=rcParams["axes.edgecolor"],
            linewidth=0,  # 1, cambiato
            clip_on=False,
        )
    return outlines_


if __name__ == '__main__':
    path = 'data_original/'
    l_freq = 0.1
    h_freq = 45

    # This dictionary has to be updated including missing subjects
    magic_numbers = {
        "s01.bdf": 105,
        "s02.bdf": 160,
        "s03.bdf": 100,
        "s04.bdf": 150,
        "s05.bdf": 150,
        "s06.bdf": 150,
        "s07.bdf": 110
    }

    subjects = get_subjects(path)
    subjects = list(magic_numbers.keys())  # Only the subjects in the dictionary
    for subject in subjects:
        raw = read_bdf(path, subject)

        montage = mne.channels.make_standard_montage('biosemi32')
        raw.set_montage(montage)

        # Filtering and then resampling avoids aliasing
        raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
        raw.resample(sfreq=128)  # così si avranno 128 MAPPE PER SECONDO

        # Computing the extrema for each subject (instead of for each trial) handles the scenario in which
        # a subject does not blink at all watching a trial:
        # their extrema would be too close to the baseline and the number of false positive blinks would be too large
        subject_lowest_peak, subject_highest_peak = get_extrema(raw, l_freq, h_freq)

        indices = get_indices_where_video_start(raw)
        for trial, index in enumerate(indices, start=1):
            # Crop the raw signal based on the 60-seconds-trial-length at given index
            cropped_raw_fp1_fp2 = crop(raw, index, FP1_FP2)
            save_plot('plots', 1, 10, cropped_raw_fp1_fp2, subject, trial)

            data = cropped_raw_fp1_fp2.get_data()
            sample_rate = get_sample_rate(cropped_raw_fp1_fp2)
            filtered_data = filter_data(data, sample_rate, l_freq=1, h_freq=10)
            trial_lowest_peak = np.min(filtered_data)
            trial_highest_peak = np.max(filtered_data)

            magic_number = magic_numbers[subject]
            thresh = calculate_thresh(filtered_data, magic_number)

            # We shall focus on Fp1 and Fp2 only since they are the closest electrodes wrt the eyes
            events = find_eog_events(cropped_raw_fp1_fp2, ch_name=FP1_FP2, thresh=thresh, l_freq=1, h_freq=10)

            # Let's save some useful information on disk
            save_events('events', subject, trial, events, index, sample_rate,
                        subject_lowest_peak, subject_highest_peak, magic_number,
                        trial_lowest_peak, trial_highest_peak,
                        thresh)

            ##########SALVATORE##########
            cropped_raw_eeg = crop(raw, index, EEG)

            idx_blinks = events[:, 0] - index
            # durata del blink prima e dopo il picco (in secondi). Può variare per soggetto/trial, meglio rimanere larghi
            blinkTime_pre = 0.09
            blinkTime_post = 0.12
            blink_pre = int(blinkTime_pre * sample_rate)
            blink_post = int(blinkTime_post * sample_rate)

            # dataTrial = cropped_raw_eeg_all.get_data()
            # dataTrial = raw.copy().pick_channels(EEG).get_data()
            rawDataset = raw.copy()
            rawEEGall = rawDataset.pick_channels(EEG)
            min1 = sample_rate * 60
            rawEEGall_trialTest = rawEEGall.crop(tmin=index / sample_rate,
                                                 tmax=(index + min1) / sample_rate)
            dataTrial = rawEEGall_trialTest.get_data()

            # crea un intorno prima e dopo il picco del blink (per l'etichettatura)
            idx_blinks_about = []
            for i in idx_blinks:
                # intorno prima del picco
                for j in range(max(0, i - blink_pre), i):  # max evita le posizioni negative
                    idx_blinks_about.append(j)

                # intorno dopo
                for j in range(i, min(i + blink_post, dataTrial.shape[1])):  # min evita le posizioni inesistenti
                    idx_blinks_about.append(j)

            # ad ogni id del picco del blink, crea un intorno con l'id precedente e successivo
            idx_blinks_near = []
            for i in idx_blinks:
                idx_blinks_near.append(i - 1)  # sample precedente
                idx_blinks_near.append(i)  # sample del picco
                idx_blinks_near.append(i + 1)  # sample successivo

            # plot topomaps
            # Monkey patches
            mne.viz.topomap._make_head_outlines = _make_head_outlines_new
            mne.viz.topomap._draw_outlines = _draw_outlines_new

            sec = 0.5
            for j in idx_blinks:
                # Ensure that  the indices are within the bounds of dataTrial
                start_index = max(j - int(sec * sample_rate), 0)
                end_index = min(j + int(sec * sample_rate), dataTrial.shape[1])
                for i in range(start_index, end_index):
                    data_sample = dataTrial[:, i].reshape(len(EEG))
                    mne.viz.plot_topomap(data=data_sample,
                                         pos=raw.info,
                                         ch_type='eeg',
                                         sensors=False,
                                         names=None,
                                         contours=0,
                                         outlines='head',
                                         sphere='eeglab',
                                         # con eeglab i sensori si trovano più esternamente rispetto a None e i sensori sono più distanziati
                                         image_interp='cubic',
                                         extrapolate='auto',
                                         border='mean',
                                         vlim=(trial_lowest_peak, trial_highest_peak),
                                         cmap='RdBu_r',
                                         # PROBLEMA RISOLTO DA SABATINA! (alcune mappe erano tutte rosse, usando RdBu_r funziona bene)
                                         cnorm=None,
                                         axes=None,
                                         show=False,
                                         onselect=None
                                         )

                    # LABELING
                    # 0 - non blink
                    # 1 - picco blink (e sample immediadamente prima e dopo)
                    # 2 - transizione da non blink a picco blink, e da picco blink a non blink
                    if i in idx_blinks_about:  # se i fa parte di un intorno di blink
                        if i in idx_blinks_near:  # se i è il picco del blink (o sample immediadamente prima o dopo)
                            label = 1  # da non usare per l'anomaly detection
                        else:  # transizione
                            label = 2  # da non usare per l'anomaly detection
                    else:  # non blink
                        label = 0

                    trialID = str(trial)
                    subject_without_extension = subject.rsplit(".", 1)[0]
                    TOPOMAPS_DIR = './topomaps_short/' + subject_without_extension + '/' + trialID + '/'

                    # If exist_ok is True, a FileExistsError is not raised if the target directory already exists
                    os.makedirs(TOPOMAPS_DIR, exist_ok=True)

                    fileName = TOPOMAPS_DIR + 'topomapSample_' + str(i) + '_blink_' + str(label) + ".png"
                    fig = plt.gcf()
                    fig.canvas.draw()
                    plt.savefig(fileName,
                                format="png",
                                dpi=41,
                                # 600 per test SOLO
                                # 41 per avere immagini 32x32
                                bbox_inches='tight',
                                pad_inches=0.01  # toglie il bianco attorno
                                )

                    plt.close()

                    print(str(i) + ' file ' + fileName + ' saved')
