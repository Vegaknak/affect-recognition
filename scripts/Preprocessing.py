from typing import Callable, Optional, Tuple, Dict, List, Any

# import libraries
import scipy.io 
import os
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
import numpy as np
import pickle
from scipy.signal import butter, filtfilt

# --------- loading & extracting data ----------

def load_AMIGOS_data(path):
    """ Loops through directory with AMIGOS data and load .mat files.
    Args: 
        Path, excluded participants
    Returns: 
        mats, tuple.
    """
    mats = []
    for i in range(40):
        pid = i + 1
        filename = f"Data_Preprocessed_P{'0' if i < 9 else ''}{pid}"
        src = os.path.join(path, filename, f"{filename}.mat")
        if os.path.exists(src):
            mats.append((pid, scipy.io.loadmat(src)))
        else:
            print(f"Warning: File not found - {src}")
    return mats

def extract_videos(mats: list, video_indices: range, excluded_ppn: list[int] = None, label: str = "video"):
    """ Extract ECG (cols 14–15) and GSR (col 16) from loaded matlab variables.
    Args:
        mats: Output of load_AMIGOS_data; dict with dataframes for each participant
        video_indices: indices of joined_data to extract
        excluded_ppn: list of participant IDs to skip for this split
        label: logging video
    Returns:
        ecg_dict, gsr_dict : dict[int, pd.DataFrame]
    """
    # convert exclusion list 
    excluded_set = set(excluded_ppn or [])
    ecg_dict: dict[int, pd.DataFrame] = {}
    gsr_dict: dict[int, pd.DataFrame] = {}

    for pid, mat in mats:
        if pid in excluded_set:  # skip if excluded 
            print(f"⏭️ Skipping participant {pid} ({label} exclusion).")
            continue

        joined = mat.get("joined_data")
        vids = mat.get("VideoIDs")
 
        ecg_segments = []  
        gsr_segments = []  

        for idx in video_indices:
            data = joined[0, idx]      # raw signal array
            vid = vids[0, idx]         # unique video identifier
            
            if not isinstance(data, np.ndarray) or data.shape[1] < 17:  # must be numpy array with ≥17 columns
                print(f"⚠️ P{pid}, {label} {idx}: invalid data.")
                continue

            n = data.shape[0]  # number of samples in segment

            # create df for ECG signals (right and left leads)
            ecg_df = pd.DataFrame({
                "ECG_Right":   data[:, 14],
                "ECG_Left":    data[:, 15],
                "VideoID":     np.full(n, vid),
                "VideoIndex":  np.full(n, idx)
            })
            ecg_segments.append(ecg_df)

            # create df for GSR signal
            gsr_df = pd.DataFrame({
                "GSR":         data[:, 16],
                "VideoID":     np.full(n, vid),
                "VideoIndex":  np.full(n, idx)
            })
            gsr_segments.append(gsr_df)

        if ecg_segments:
            ecg_dict[pid] = pd.concat(ecg_segments, ignore_index=True)
        if gsr_segments:
            gsr_dict[pid] = pd.concat(gsr_segments, ignore_index=True)

    return ecg_dict, gsr_dict     # Return two dicts mapping pid → df

def _load_signal(file_path: str, vid_num: int, col_name: str) -> pd.DataFrame:
    """
    Read an EDA or BVP CSV from Empatica E4 and return a long-format
    df with columns  ['VideoID', col_name].

    E4 CSVs have no header row, the first line is a timestamp,
       the second line the sampling frequency, the rest is the signal.
    They also store several samples per row, which are flattened
    """
    raw = pd.read_csv(file_path, header=None)
    signal_only = raw.iloc[2:, :] # drop the first two metadata rows (timestamp & freq)
    flat = (                      # flatten the remaining values and drop NaNs
        signal_only.values                   
        .astype(float)                        
        .ravel(order="C")                     
    )
    flat = flat[~pd.isna(flat)]              # remove NaNs

    return pd.DataFrame(
        {
            "VideoID": vid_num,
            col_name: flat,
        }
    )

def load_phymer_data(root_dir):
    """Return two dicts indexed by **numeric participant id**:
        eda_data[pid]  -> DataFrame ['VideoID', 'EDA_raw']
        bvp_data[pid]  -> DataFrame ['VideoID', 'BVP_raw']
    """
    eda_data: Dict[int, pd.DataFrame] = {}
    bvp_data: Dict[int, pd.DataFrame] = {}

    e4_root = os.path.join(root_dir, "phyMER Dataset", "E4")

    for subj in sorted(os.listdir(e4_root)):
        subj_path = os.path.join(e4_root, subj)
        if not os.path.isdir(subj_path):
            continue

        try:                            # e.g. "Sub01" → 1 
            pid = int("".join(filter(str.isdigit, subj)))
        except ValueError:
            print(f"Cannot parse participant id from «{subj}» – skipped.")
            continue

        eda_parts, bvp_parts = [], []

        for vid_dir in sorted(os.listdir(subj_path)):
            vid_path = os.path.join(subj_path, vid_dir)
            if not os.path.isdir(vid_path):
                continue

            try:                        # "SUB01VID07" → 7
                vid_num = int("".join(filter(str.isdigit, vid_dir.split("VID")[-1])))
            except ValueError:
                vid_num = vid_dir      

            base = vid_dir
            eda_file = os.path.join(vid_path, f"{base}_EDA_4.csv")
            bvp_file = os.path.join(vid_path, f"{base}_BVP_64.csv")

            missing = [s for s, f in [("EDA", eda_file), ("BVP", bvp_file)]
                       if not os.path.isfile(f)]
            if missing:
                print(f"! Missing {', '.join(missing)} for participant {pid}, video {vid_num}")
                continue

            eda_parts.append(_load_signal(eda_file, vid_num, "EDA_raw"))
            bvp_parts.append(_load_signal(bvp_file, vid_num, "BVP_raw"))

        if eda_parts:
            eda_data[pid] = pd.concat(eda_parts, ignore_index=True)
        if bvp_parts:
            bvp_data[pid] = pd.concat(bvp_parts, ignore_index=True)

    return eda_data, bvp_data


# --------- preprocessing steps ----------
def convert_gsr(GSR_data):
    """ Converts GSR data from Ohms to microSiemens and clip unrealistic values
    Args: 
        GSR_data (dict): Dictionary of DataFrames with a 'GSR' column (in Ohms).
    Returns: 
        Same structure (dict) but with a new column with the converted GSR response
        'GSR_uS' (in µS), and invalid (≤ 0) values removed.
    """
    
    for ppn, df in GSR_data.items():
        df = df[df["GSR"] > 0].copy()  # filter and avoid SettingWithCopyWarning. remove corrupt/negative values
        df["GSR_uS"] = 1e6 / df["GSR"]  # convert
        GSR_data[ppn] = df  # update with cleaned + converted dataframe
    return GSR_data

def winsorize_signal(df, col="GSR_uS", limits=(0.01, 0.01), group_col="VideoID"):
    """ Winsorizes values in the specified biosignal column per group.
    Args:
        df (pd.DataFrame): The input dataframe.
        col (str): Name of the column to winsorize.
        group_col (str): Column to group by.
        limits (tuple): Lower and upper percentile limits.
    Returns:
        pd.Series: Winsorized version of the signal.
    """
    winsorized = pd.Series(index=df.index, dtype=float)
    
    for group_val, group_df in df.groupby(group_col): # make sure we group by videoID, so that we winsorize per recording (and not over all recordings)
        original = group_df[col].values.copy()
        clipped = winsorize(original, limits=limits)
        winsorized.loc[group_df.index] = clipped
    return winsorized

def butterworth_filter(col, sampling_rate, method="neurokit", signal_type=None):
    """ Clean biosignal using NeuroKit2. 
    Args:
        col (array-like): Raw biosignal column.
        sampling_rate (int): Sampling rate in Hz.
        method (str): Filtering method (used by NeuroKit2).
        signal_type (str): "ecg" or "gsr".
    Returns:
        array-like: Cleaned signal.
    """
    if signal_type.lower() == "gsr":
        return nk.eda_clean(col, sampling_rate, method)
    elif signal_type.lower() =="ecg":
        return nk.ecg_clean(col, sampling_rate, method="biosppy")
    elif signal_type.lower() == "ppg":
        return nk.ppg_clean(col, sampling_rate, method)
    else:
        print(f"Invalid signal type: {signal_type}")
        
# different filtering for phymer
def butter_lowpass(x, fs=4, fc=0.45, order=2):
    wn = fc / (fs / 2)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x)          # zero-phase

def apply_butterworth(sig_dict, in_col="EDA_raw", out_col="EDA_filt",
                      fs=4, fc=0.45, order=2):
    """Add a low-passed column to every DataFrame in the dict."""
    for df in sig_dict.values():
        df[out_col] = butter_lowpass(df[in_col].values,
                                     fs=fs, fc=fc, order=order)
    return sig_dict


def within_subj_normalize(data_dict, signal_col, new_col=None):
    """ Normalizes a filtered signal column per participant using within-participant z score normalization

    Params:
        data_dict: dict {participant_id: DataFrame}
        signal_col: str, column name to normalize (e.g. "GSR_filtered")
        new_col: str or None, if given it will create new col with normalized values. If None, replaces signal_col
    """
    normalized_dict = {}
    for pid, df in data_dict.items():
        x = df[signal_col]
        x_norm = (x - x.mean()) / x.std()
        df_copy = df.copy()
        if new_col:
            df_copy[new_col] = x_norm
        else:
            df_copy[signal_col] = x_norm
        normalized_dict[pid] = df_copy
    return normalized_dict


def plot_gsr(df, col="GSR_z", group_col="VideoIndex", participant_id=None, sample_range=(0, 500)):
    """ Plots the GSR signal per video in separate figures, optionally zoomed in.
    Args:
        df (pd.DataFrame): Participant GSR DataFrame.
        col (str): Column to plot (e.g., "GSR_z", "GSR_filtered", or "GSR_uS").
        group_col (str): Column to group by per video (e.g., "VideoIndex" or "VideoID").
        participant_id (int or None): Optional; adds participant ID to the title.
        sample_range (tuple or None): Optional (start, end) sample indices to zoom in.
    """
    videos = sorted(df[group_col].unique(), key=str)
    for vid in videos:
        segment = df[df[group_col] == vid]
        if sample_range:
            segment = segment.iloc[sample_range[0]:sample_range[1]]

        plt.figure(figsize=(10, 3))
        plt.plot(segment[col].values, linewidth=0.8)
        title = f"GSR {col} - Video {vid}"
        if participant_id is not None:
            title = f"Participant {participant_id} - {title}"
        plt.title(title)
        plt.xlabel("Samples")
        plt.ylabel("Filtered GSR in uS")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
def plot_ecg(ecg_data, participant_id, sample_range=None):
    """ Plots ECG (Right and Left) signals per video for a given participant.

    Parameters:
        ecg_data (dict): Dictionary of ECG DataFrames per participant.
        participant_id (int): Participant ID to plot.
        sample_range (tuple or None): (start, end) indices to trim each video segment.
    """
    df = ecg_data.get(participant_id)
    if df is None:
        print(f"Participant {participant_id} not found in ECG data.")
        return

    grouped = df.groupby("VideoID")
    n = len(grouped)

    fig, axs = plt.subplots(n, 1, figsize=(12, 2.5 * n), squeeze=False)
    axs = axs.flatten()

    for ax, (vid, segment) in zip(axs, grouped):
        right = segment["ECG_Right_z"] # CHANGE BACK TO ECG_Right_z
        left = segment["ECG_Left_z"] # CHANGE BACK TO ECG_Left_z
        if sample_range:
            right = right.iloc[sample_range[0]:sample_range[1]]
            left = left.iloc[sample_range[0]:sample_range[1]]

        ax.plot(right.values, label="ECG_Right", alpha=0.9)
        ax.plot(left.values, label="ECG_Left", alpha=0.6)
        ax.set_title(f"Video {vid} - Participant {participant_id}")
        ax.set_ylabel("ECG Amplitude")
        ax.grid(True)
        ax.legend()

    axs[-1].set_xlabel("Sample Index")
    plt.tight_layout()
    plt.show()

def decompose_eda(data_dict, sampling_rate=128, method="highpass", col="GSR_z"):
    """ Extracts tonic and phasic components from a dictionary of cleaned GSR data. 
            Possibly, for recordings longer than threshold_sec (15 minutes by default), a sliding-window decomposition is applied.
    
    Args:
        data_dict: dict where each key is a participant ID and each value is a DataFrame with at least two columns: "GSR_z" (the cleaned signal) and "VideoID".
        sampling_rate: Sampling frequency in Hz (default 128).
        method: Method used by nk.eda_phasic (default "highpass").
        col: column you want to decompose
    Returns
        decomposition_dict (dict): Nested dictionary with seperated tonic and phasic activity
    """
    new_dict = {}
    
    for pid, df in data_dict.items():
        new_df_list = []
        for video_id, group in df.groupby("VideoID"):
            cleaned_signal = group[col].values # CHANGE BACK TO GSR_Z
            N = len(cleaned_signal)
            
            # Global decomposition (no sliding window used).
            decomposition = nk.eda_phasic(cleaned_signal, sampling_rate=sampling_rate, method=method)
            tonic = decomposition["EDA_Tonic"].values
            phasic = decomposition["EDA_Phasic"].values
            time_axis = np.arange(N)
            
            # Create a copy of the group and add the new columns.
            group = group.copy()
            group["Tonic"] = tonic
            group["Phasic"] = phasic
            group["Time"] = time_axis
            new_df_list.append(group)
        
        # Concatenate all video groups back into a single DataFrame for the participant.
        new_dict[pid] = pd.concat(new_df_list, axis=0).reset_index(drop=True)
    
    return new_dict

def extract_phasic_features(phasic, info, time_sec):
    """ Extract phasic (SCR) features
    Args:
        phasic: array with the phasic signal.
        info: dict from nk.eda_peaks() containing keys "SCR_Amplitude", "SCR_RiseTime", "SCR_RecoveryTime", etc.
        time_sec: time vector in seconds.
    Returns: dict with phasic gsr features.
    """
    # SCR amplitudes
    scr_amplitudes = info.get("SCR_Amplitude") # SCR_Amplitude should contain amplitude values of each detected SCR (phasic peak)
    scr_amplitudes = scr_amplitudes[~np.isnan(scr_amplitudes)]
    # mean and sd of amplitude
    mean_amp = np.mean(scr_amplitudes) if len(scr_amplitudes) > 0 else 0 # calculates mean avg amplitude of all detected SCRs. If no peaks detected, put in a 0 for mean.
    sd_amp = np.std(scr_amplitudes, ddof=1) if len(scr_amplitudes) > 1 else 0 # check how many NS-SCRS there are after filtering nans. if there are at least 2, compute SD. if there are less then 2, put 0.
    
    # NS-SCR Frequency
    duration_min = time_sec[-1] / 60 if time_sec[-1] > 0 else 1 # total recording duration in minutes, convert back to seconds
    ns_frequency = len(scr_amplitudes) / duration_min # number of SCRs detected divided by duration in minutes. 

    # mean rise time -> ignore NaNs in SCR_RiseTime
    rise_times     = np.array(info.get("SCR_RiseTime", []), dtype=float)# nk.peaks() returns rise times if able to find a clean/valid onset-to-peak transition for each SCR event. 
    valid_rises    = rise_times[~np.isnan(rise_times)]
    mean_rise      = valid_rises.mean() if valid_rises.size > 0 else np.nan
    # mean_rise = np.mean(rise_times) if (rise_times is not None and len(rise_times) > 0) else np.nan # average time for SCR to rise from onset to peak. if recovery was too noisy, cut off, or did not return to baseline, return nan

    # recovery times (default is half recovery time, but I set it to .25)
    recovery_time = info.get("SCR_RecoveryTime") # get (half) recovery times from eda_peaks
    # cap at 12 s to avoid large values due to sensor measurement
    if recovery_time is not None and np.any(~np.isnan(recovery_time)):
        true_recovery = recovery_time[recovery_time < 12]
        mean_recovery = np.nanmean(true_recovery) if len(true_recovery) > 0 else np.nan
    else:
        mean_recovery = np.nan     # np.nan() ignores nans


    return {
        "MeanSCRAmplitude": mean_amp,
        "SDSCRAmplitude": sd_amp,
        "NSSCRFrequency": ns_frequency,
        "MeanRiseTime": mean_rise,
        "MeanRecoveryTime_50": mean_recovery,
    }

def extract_tonic_features(tonic, time_sec):
    """ Extract tonic (SCL) features.
    Args: 
        tonic (array_like): The tonic signal.
        time_sec (array_like): Time vector in seconds.
    Returns (dict): Features including mean SCL and the slope of the tonic signal.
    """
    
    mean_scl = np.mean(tonic) # takes mean of all values in tonic component
    slope = np.polyfit(time_sec, tonic, 1)[0] if len(time_sec) > 1 else np.nan # fit line to data (y=tonic, x=time) and extract slope
    return {"MeanSCL": mean_scl, "SCLSlope": slope}

#  extractor for a whole decomposed dictionary
def eda_feature_table(decomposed_dict, *, sampling_rate=128,
                           peak_method="kim2004", amplitude_min=0.1):
    """ Executes feature extraction and makes a table
    Args: 
        decomposed_dict (dict {participant_id: DataFrame with columns ["Phasic", "Tonic", "Time", "VideoID"]}
        sampling_rate   : int
        peak_method     : str  → passed to nk.eda_peaks
    Returns
        pd.DataFrame   one row per (ParticipantID, VideoID) with all features
    """
    rows = []

    for pid, df in decomposed_dict.items():
        for vid, vdf in df.groupby("VideoID"):
            phasic = vdf["Phasic"].values
            tonic  = vdf["Tonic"].values
            time_s = vdf["Time"].values / sampling_rate

            try:
                _, info = nk.eda_peaks(phasic,
                                       sampling_rate=sampling_rate,
                                       method=peak_method,
                                       amplitude_min=amplitude_min)

                row = {
                    "ParticipantID": pid,
                    "VideoID":      vid,
                    **extract_phasic_features(phasic, info, time_s), # call functions from earlier
                    **extract_tonic_features(tonic,  time_s) 
                }
                rows.append(row)

            except Exception as err:
                print(f"P{pid}, video {vid}: {err}")

    return pd.DataFrame(rows)

def plot_eda_decomposed(decomposition_dict, participant, video_id, sampling_rate=128, col="GSR_z"):
    """ Plots the cleaned EDA signal along with its tonic and phasic components for a specified participant and video.
    Optionally, it computes and overlays the EDA peaks as detected by nk.eda_peaks().
    
    Args:
        decomposition_dict (dict): Nested dictionary returned by decompose_eda(), expected to contain
            columns including "Time", "cleaned" (or "GSR_z"), "Tonic", and "Phasic".
        participant (int or str): Participant ID (key in the dictionary).
        video_id (str): Video ID (sub-key corresponding to the participant).
        sampling_rate (int): Sampling rate of the signals. Default is 128.
        show_peaks (bool): Whether to compute and overlay EDA peaks. Default is True.
    """
    
    # Retrieve the data for the given participant and video.
    try:
        df = decomposition_dict[participant]
        data = df[df["VideoID"] == video_id]  # Match on video ID.
        if data.empty:
            print(f"No data found for participant {participant} and video {video_id}.")
            return
    except KeyError:
        print(f"No data found for participant {participant}.")
        return

    time = data["Time"].values
    # Use the cleaned signal; set col to the one you want plotted
    cleaned_signal = data["cleaned"].values if "cleaned" in data.columns else data[col].values
    tonic = data["Tonic"].values
    phasic = data["Phasic"].values

    plt.figure(figsize=(10, 6))
    plt.plot(time, cleaned_signal, label="Cleaned Signal", color="gray", alpha=0.6)
    plt.plot(time, tonic, label="Tonic Component", color="blue")
    plt.plot(time, phasic, label="Phasic Component", color="red")
    

    # Compute EDA peaks using NeuroKit2.
    eda_peaks, info_peaks = nk.eda_peaks(phasic, sampling_rate=sampling_rate, amplitude_min=0.1)
    # The returned dictionary contains a binary column "EDA_Peaks" with 1's marking detected peaks.
    peak_indices = np.where(eda_peaks["SCR_Peaks"] == 1)[0]
    if len(peak_indices) > 0:
        plt.scatter(time[peak_indices], phasic[peak_indices],
                    color="orange", marker="o", label="Detected Peaks", zorder=3)
    else:
        print("No EDA peaks detected.")
    
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title(f"Participant {participant} - Video {video_id}")
    plt.legend()
    plt.tight_layout()
    plt.show()
    

# extract BVP features:

def bvp_feature_table(signal_dict, sampling_rate=64, peak_method="elgendi"):
    """ Extract BVP/PPG features per participant×video.
    Params:
        signal_dict: dict[int, pd.DataFrame], each df contains ["VideoID","BVP_z"].
        sampling_rate (int) in Hz
        peak_method (str): peak detection method used by Neurokit
    Returns:
        dataframe with columns: ParticipantID, VideoID, MeanIBI, MeanHR, SDNN, RMSSD, pNN50
    """
    rows = []
    for pid, df in signal_dict.items():
        for vid, segment in df.groupby("VideoID"):
            sig = segment["BVP_z"].values

            # Peak detection
            peaks_df, info = nk.ppg_peaks(
                sig,
                sampling_rate=sampling_rate,
                method=peak_method,
                correct_artifacts=True
            )
            peak_idx = peaks_df.index[peaks_df["PPG_Peaks"] == 1].to_numpy()

            # Compute IBIs (s)
            times = peak_idx / sampling_rate
            ibi   = np.diff(times)

            # 3) Features
            if len(ibi) > 0:
                mean_ibi = np.mean(ibi)
                mean_hr  = 60 / mean_ibi                       # bpm from mean IBI
            else:
                mean_ibi = np.nan
                mean_hr  = np.nan

            if len(ibi) > 1: # to calculate hrv, we need at least two ibi's
                sdnn   = np.std(ibi,  ddof=1) * 1000
                rmssd  = np.sqrt(np.mean(np.diff(ibi)**2)) * 1000 
                pnn50  = np.sum(np.abs(np.diff(ibi)) > 0.05) / len(ibi) * 100
            else:
                sdnn = rmssd = pnn50 = np.nan

            rows.append({
                "ParticipantID": pid,
                "VideoID":       vid,
                "MeanIBI":       mean_ibi,
                "MeanHR":        mean_hr,
                "SDNN":          sdnn,
                "RMSSD":         rmssd,
                "pNN50":         pnn50
            })

    return pd.DataFrame(rows)

def extract_ecg_features(ecg_signal, sampling_rate=128):
    """
    Extracts time-domain HRV and morphological intervals (QRS, PR, QT)
    from an already-cleaned ECG signal computed over the entire recording.
    
    Returns a dictionary with features: mean HR (HR_mean), SDNN, RMSSD, pNN50,
    Mean_IBI (average NN interval in seconds), and the average durations of the 
    QRS complex, PR interval, and QT interval (all in seconds).
    """
    # Convert the signal to a numpy array.
    ecg_signal = np.array(ecg_signal)
    
    # 1) Detect R-peaks in the entire signal.
    peaks, info = nk.ecg_peaks(
        ecg_signal,
        sampling_rate=sampling_rate,
        method="neurokit",          # or "pantompkins1985"
        correct_artifacts=True,     # <- re-fills missed beats, drops extras
    )
    # Use the indices from the info dictionary.
    rpeaks_indices = info["ECG_R_Peaks"]
    
    # 2) Compute time-domain HRV metrics over the whole recording.
    # hrv_time = nk.hrv_time(ecg_peaks, sampling_rate=sampling_rate, show=False)
    hrv_time = nk.hrv_time(rpeaks_indices, sampling_rate=sampling_rate, show=False)

    
    # 3) Perform wave delineation using the extracted R-peak indices.
    signals_delineate, waves_delineate = nk.ecg_delineate(
        ecg_signal,
        rpeaks=rpeaks_indices,
        sampling_rate=sampling_rate,
        method="dwt",
        show=False
    )
    
    # Compute mean heart rate from HRV_MeanNN (which is in ms).
    meanNN_sec = hrv_time["HRV_MeanNN"][0] / 1000.0  
    mean_heart_rate = 60.0 / meanNN_sec if (not np.isnan(meanNN_sec) and meanNN_sec > 0) else np.nan

    # === Compute QRS Duration (from Q-peak to S-peak) ===
    qrs_durations = []
    if "ECG_Q_Peaks" in waves_delineate and "ECG_S_Peaks" in waves_delineate:
        q_peaks = waves_delineate["ECG_Q_Peaks"]
        s_peaks = waves_delineate["ECG_S_Peaks"]
        # Use minimum length if arrays differ.
        for q, s in zip(q_peaks, s_peaks):
            if not np.isnan(q) and not np.isnan(s) and s > q:
                qrs_durations.append((s - q) / sampling_rate)
    qrs_mean = np.mean(qrs_durations) if qrs_durations else np.nan

    # === Compute PR Interval (from P onset to R peak) ===
    pr_intervals = []
    if "ECG_P_Onsets" in waves_delineate:
        p_onsets = waves_delineate["ECG_P_Onsets"]
        # For each R-peak, find the most recent P onset before the R peak.
        for r in rpeaks_indices:
            valid_p = [p for p in p_onsets if p < r]
            if valid_p:
                p_val = max(valid_p)
                pr_intervals.append((r - p_val) / sampling_rate)
    pr_mean = np.mean(pr_intervals) if pr_intervals else np.nan

    # === Compute QT Interval (from Q-peak to T offset) ===
    qt_intervals = []
    if "ECG_T_Offsets" in waves_delineate and "ECG_Q_Peaks" in waves_delineate:
        t_offsets = waves_delineate["ECG_T_Offsets"]
        q_peaks = waves_delineate["ECG_Q_Peaks"]
        for q, t in zip(q_peaks, t_offsets):
            if not np.isnan(q) and not np.isnan(t) and t > q:
                qt_intervals.append((t - q) / sampling_rate)
    qt_mean = np.mean(qt_intervals) if qt_intervals else np.nan

    # Build dictionary for storing features
    features_dict = {
        "HR_mean": mean_heart_rate,       # beats per minute
        "SDNN": hrv_time["HRV_SDNN"][0],    # in ms
        "RMSSD": hrv_time["HRV_RMSSD"][0],    # in ms
        "pNN50": hrv_time["HRV_pNN50"][0],    # in percent
        "Mean_IBI": meanNN_sec,             # in seconds
        "QRS_mean": qrs_mean,               # average QRS duration (s)
        "PR_mean": pr_mean,                 # average PR interval (s)
        "QT_mean": qt_mean                  # average QT interval (s)
    }
    return features_dict

# helper functions for extractig features
def compute_features_for_dictionary(ecg_dict, sampling_rate=128):
    """ Given a dictionary of DataFrames (with key as person and value as the DataFrame), 
    computes ECG features for each trial (grouped by VideoID) over the entire recording.
    
    Returns a single DataFrame with columns: pid, VideoId, and all feature columns.
    """
    all_results = []

    for person_id, df_person in ecg_dict.items():
        # Group by VideoID for each trial.
        grouped = df_person.groupby("VideoID")
        for video_id, df_video in grouped:
            # Use the appropriate channel, e.g., "ECG_Right_z".
            ecg_signal = df_video["ECG_Right_z"].values # CHANGE BACK TO ECG_right_z -> use unscaled for delineation?
            feats = extract_ecg_features(ecg_signal, sampling_rate=sampling_rate)
            # Add identifier columns.
            feats["ParticipantID"] = person_id
            feats["VideoID"] = video_id
            all_results.append(feats)

    final_df = pd.DataFrame(all_results)
    return final_df

# helper functions for preprocessing steps into dicts
def winsorize_dict(sig_dict, col, limits, by="VideoID"):
    for df in sig_dict.values():
        df[col] = winsorize_signal(df, col=col, limits=limits, group_col=by)
    return sig_dict

def butterworth_dict(sig_dict, col, new_col, sig_type, fs):
    for df in sig_dict.values():
        df[new_col] = butterworth_filter(df[col], sampling_rate=fs, signal_type=sig_type)
    return sig_dict


def exp_condition(df, session_df):
    """ Merge in session type and add flag for if person watched the long videos alone or with people.
    Args: main long df and df with session type (metadata). In the metadata, there is a column called 'Session_Type_Exp_2' that tracks 'Alone' or 'Group'
    Returns: df with new column that holds a binary indicator for 1 (alone) or 0 (group)
    """
    # ensure the merge keys share type
    df = df.copy()
    df['ParticipantID'] = df['ParticipantID'].astype(str)
    session_df = session_df.copy()
    session_df['UserID'] = session_df['UserID'].astype(str)

    # extract and rename
    flags = session_df[['UserID', 'Session_Type_Exp_2']].copy()
    flags.rename(columns={'UserID': 'ParticipantID'}, inplace=True)
    flags['Alone_long'] = (flags['Session_Type_Exp_2'] == 'Alone').astype(int)

    # now merge on ParticipantID (both strings)
    out = df.merge(flags[['ParticipantID', 'Alone_long']],
                   on='ParticipantID', how='left')
    return out

# ----------- execution functions -----------

# ---------------------------------------------------------------
#  AMIGOS preprocessing + feature-extraction pipeline
# ---------------------------------------------------------------
def run_pipeline(
    ecg_s,
    gsr_s,
    ecg_l, 
    gsr_l, 
    winsor_limits: tuple[float, float] = (0.01, 0.01), 
    fs: int = 128, 
    save_csv = None):
    """ Full preprocessing, feature extraction and running the models for short and long AMIGOS data.
    Returns merged dataframes containing all features per participant, trial.
    """

    # ------ PREPROCESSING -------
    # Winsorize 
    gsr_s = winsorize_dict(gsr_s, "GSR_uS", winsor_limits)
    gsr_l = winsorize_dict(gsr_l, "GSR_uS", winsor_limits)
    ecg_s = winsorize_dict(ecg_s, "ECG_Right", (0.01,0.01))
    ecg_l = winsorize_dict(ecg_l, "ECG_Right", (0.01,0.01))

    # Filter
    gsr_s = butterworth_dict(gsr_s, "GSR_uS", "GSR_filtered", "gsr", fs)
    gsr_l = butterworth_dict(gsr_l, "GSR_uS", "GSR_filtered", "gsr", fs)
    ecg_s = butterworth_dict(ecg_s, "ECG_Right", "ECG_Right_filtered", "ecg", fs)
    ecg_l = butterworth_dict(ecg_l, "ECG_Right", "ECG_Right_filtered", "ecg", fs)

    # Normalize (within subj)
    gsr_s = within_subj_normalize(gsr_s, "GSR_filtered", "GSR_z")
    gsr_l = within_subj_normalize(gsr_l, "GSR_filtered", "GSR_z")
    ecg_s = within_subj_normalize(ecg_s, "ECG_Right_filtered", "ECG_Right_z")
    ecg_l = within_subj_normalize(ecg_l, "ECG_Right_filtered", "ECG_Right_z")

    # select only z_scored column (perhaps unnecesary)
    gsr_s_clean = {pid: df[["GSR_z","VideoID"]] for pid, df in gsr_s.items()}
    gsr_l_clean = {pid: df[["GSR_z","VideoID"]] for pid, df in gsr_l.items()}
    ecg_s_clean = {pid: df[["ECG_Right_z","VideoID"]] for pid, df in ecg_s.items()}
    ecg_l_clean = {pid: df[["ECG_Right_z","VideoID"]] for pid, df in ecg_l.items()}

    # Decompose eda
    decomp_s = decompose_eda(gsr_s_clean, sampling_rate=fs)
    decomp_l = decompose_eda(gsr_l_clean, sampling_rate=fs)

    # ------ FEATURE EXTRACTION -------
    gsr_s_feats = eda_feature_table(decomp_s, sampling_rate=fs)
    gsr_l_feats = eda_feature_table(decomp_l, sampling_rate=fs)
    ecg_s_feats = compute_features_for_dictionary(ecg_s_clean, sampling_rate=fs)
    ecg_l_feats = compute_features_for_dictionary(ecg_l_clean, sampling_rate=fs)

    # drop outlier trials - only relevant for ECG
    n_short_before = len(ecg_s_feats)
    n_long_before  = len(ecg_l_feats)

    ecg_s_feats = ecg_s_feats[ecg_s_feats["SDNN"] <= 300].reset_index(drop=True)
    ecg_l_feats = ecg_l_feats[ecg_l_feats["SDNN"] <= 300].reset_index(drop=True)

    n_short_after = len(ecg_s_feats)
    n_long_after  = len(ecg_l_feats)

    print(f"Short: removed {n_short_before - n_short_after} trials with SDNN > {300} ms.")
    print(f"Long: removed {n_long_before - n_long_after} trials with SDNN > {300} ms.")
        
    # Merge ecg and gsr features
    merged_short = pd.merge(gsr_s_feats, ecg_s_feats, on=["ParticipantID","VideoID"], how="inner")
    merged_long  = pd.merge(gsr_l_feats, ecg_l_feats, on=["ParticipantID","VideoID"], how="inner")

    if save_csv is not None:
        short_path, long_path = save_csv
        merged_short.to_pickle(short_path)
        merged_long.to_pickle(long_path)
        print(f"Saved features to:\n  - {short_path}\n  - {long_path}")

    return merged_short, merged_long

# ---------------------------------------------------------------
#  PhyMER preprocessing + feature-extraction pipeline
# ---------------------------------------------------------------
def run_phymer_pipeline(
    root_dir: str,
    *,
    fs_eda: int           = 4,         # Empatica EDA sampling-rate
    fs_bvp: int           = 64,        # Empatica BVP sampling-rate
    bpf_fc: float         = 0.45,       # low-pass cut-off for EDA (Hz)
    bpf_order: int        = 2,         # Butterworth order
    wins_limits: tuple    = (0.01, 0.01),  # winsor-tails 
    amp_min: float        = 0.1,      # amplitude_min for SCR peak det.
    peak_method_eda: str  = "neurokit",
    peak_method_bvp: str  = "elgendi",
    save_csv  = "PHY_features.csv"
):
    """ End-to-end preprocessing for the PhyMER dataset.
    Parameters:
    root_dir : str
        Folder that contains the raw PhyMER sub-directories.
    fs_eda, fs_bvp : int
        Native sampling rates of the EDA and BVP channels.
    bpf_fc, bpf_order : float, int
        Parameters of the Butterworth low-pass used for EDA denoising.
    wins_limits : tuple(float,float)
        Lower / upper tails for winsorisation.
    amp_min : float
        `amplitude_min` forwarded to `nk.eda_peaks`.
    peak_method_eda, peak_method_bvp : str
        Which detector back-end to use in NeuroKit.
    save_csv : str | None
          • path → save merged feature table to disk  
          • None → skip saving

    Returns
    combined_features : pd.DataFrame
        One row per (ParticipantID, VideoID) containing all extracted
        EDA + BVP features.
    """
    # ------ LOAD -------
    eda_dict, bvp_dict = load_phymer_data(root_dir)

    # ------ PREPROCESSING -------
    # winsorize
    eda_dict = winsorize_dict(eda_dict, col="EDA_raw", limits=wins_limits, by="VideoID")
    bvp_dict = winsorize_dict(bvp_dict, col="BVP_raw", limits=(0.03, 0.03), by="VideoID")

    # filter (butterworth)
    eda_dict = apply_butterworth(eda_dict, in_col="EDA_raw", out_col="EDA_filt", fs=fs_eda, fc=bpf_fc, order=bpf_order)
    bvp_dict = butterworth_dict(bvp_dict, col="BVP_raw", new_col="BVP_filtered", sig_type="ppg", fs=fs_bvp)

    # Within-subject z-scaling 
    eda_dict = within_subj_normalize(eda_dict, signal_col="EDA_filt", new_col="EDA_z")
    bvp_dict = within_subj_normalize(bvp_dict, signal_col="BVP_filtered", new_col="BVP_z")

    # ------ FEATURE EXTRACTION -------
    # Decompose EDA + extract features 
    decomposed   = decompose_eda(eda_dict, col="EDA_z", sampling_rate=fs_eda)
    eda_features = eda_feature_table(decomposed, sampling_rate = fs_eda, peak_method   = peak_method_eda,  amplitude_min = amp_min)

    # extract features for bvp
    bvp_features = bvp_feature_table(bvp_dict, sampling_rate = fs_bvp, peak_method = peak_method_bvp)

    # Merge feature tables (eda+bvp)
    combined = pd.merge(
        eda_features,
        bvp_features,
        on=["ParticipantID", "VideoID"],
        how="inner",
        suffixes=("_eda", "_ppg")
    )

    # drop unrealistic trials 
    if "RMSSD" in combined.columns:
        before_rows = len(combined)
        combined = combined[combined["RMSSD"] <= 300].copy()
        after_rows = len(combined)
        print(f"Dropped {before_rows - after_rows} trials with RMSSD > {300} ms")
    else:
        print("RMSSD column not found — skipping RMSSD filtering")

    # save to csv
    if save_csv is not None:
        combined.to_csv(save_csv, index=False)
        print(f"✔ Combined feature set saved   →  {save_csv}")

    return combined

   