from clc_analysis.phase.actual_phase import actual_phase
from scipy import signal
import numpy as np
import pandas as pd
# from pingouin import circ_r
from clc_analysis.phase.filter_DIO import get_sensor_on_period


# def closest_indices(a, b):
#     indices = np.searchsorted(b, a)
#     left = np.take(b, np.maximum(indices - 1, 0), mode='clip')
#     right = np.take(b, np.minimum(indices, len(b) - 1), mode='clip')
#     closest = np.where(np.abs(left - a) < np.abs(right - a), indices - 1, indices)
#     return closest


def closest_indices(a, b):
    """
    Find indices of the closest elements in b for each element in a.

    Parameters:
    a (array-like): First array.
    b (array-like): Second array.

    Returns:
    np.ndarray: Array of indices of the closest elements in b for each element in a.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    
    # Compute the pairwise differences
    diff = np.abs(a[:, np.newaxis] - b)
    
    # Find the indices of the minimum differences
    indices = np.argmin(diff, axis=1)
    
    return indices


def find_phase_resets(phase):
    prev_phase = -1
    reset_idx = []
    for i, cur_phase in enumerate(phase):
        if (prev_phase-cur_phase) > np.pi:
            reset_idx.append(i)
        prev_phase= cur_phase
    return np.array(reset_idx)


def organize_cycles(
    SessionData = None,
    raw_in = None, 
    time_in = None, 
    timestamp_in = None, 
    estimated_phase_in = None,
    filter_lowcut = None, 
    filter_highcut = None, 
    fs_filter = None,
    event_idx = None, 
    event_time = None, 
    event_timestamp = None, 
    event_length = None,
    label_in_reward=False, 
    reward_sensor_name=None,
    label_after_first_event=False,
    label_above_threshold=False, 
    amplitude_threshold=None, 
    use_peak=False
    ):
    """
    Organize and group the input data types by oscillatory cycles, and label the cycles.

    Parameters
    ----------
    SessionData : SessionData class object, should at least have raw LFP data to define cycles
    raw_in: array of raw signal data, derive filtered data, envelope and phase from it if SessionData not given
    time_in: array of signal time, used if SessionData not given
    timestamp_in: array of signal timestamp, used if SessionData not given
    estimated_phase_in: online estimated phase, used if SessionData not given
    filter_lowcut:
    filter_highcut:
    fs_filter:
    target : array of targets that events might align with, if any
    target_index : array of the corresponding indices of targets in LFP/time/timestamp array (import from trodes)
    event_idx : array of event time indices
    event_time:
    event_timestamp:
    event_length : array of event duration corresponding to event_time
    label_in_reward: boolean, whether to label a cycle as during reward consumption or not
    reward_sensor_name: array like, array of strings (reward sensor names)
    label_after_first_event: boolean, whether to label a cycle as happening after the first stim or not
    label_above_threshold: boolean, whether to label a cycle as above power threshold or not
    amplitude_threshold: float
    use_peak: boolean, whether to use peaks as beginning and end of cycles

    Returns
    -------
    df_cycles: pandas dataframe in which each row contains data in an oscillatory cycle
    """
    if SessionData is not None:
        raw = SessionData.raw
        filtered = SessionData.filtered
        phase = SessionData.phase
        lin_phase = np.unwrap(phase)
        time = SessionData.time
        timestamp = SessionData.timestamp
        estimated_phase = SessionData.estimated_phase
        envelope = SessionData.envelope

    else:
        raw = raw_in
        timestamp = timestamp_in
        if time_in is not None:
            time = time_in
        else:
            time = (timestamp - timestamp[0]) / fs_filter
        butter_filter = signal.butter(2, [filter_lowcut, filter_highcut], 'bp', fs=fs_filter, output='sos')
        filtered = signal.sosfiltfilt(butter_filter, raw)
        analytical = signal.hilbert(filtered)
        envelope = np.abs(analytical)
        phase = np.angle(analytical)
        lin_phase = np.unwrap(phase)
        estimated_phase = estimated_phase_in

    if event_time is None and event_idx is not None:
        event_time = time[event_idx]

    if event_timestamp is None and event_idx is not None:
        event_timestamp = timestamp[event_idx]

    if use_peak:
        end_index, _ = signal.find_peaks(filtered, height=0)
    else:
        end_index, _ = signal.find_peaks(-filtered, height=0)

    # index arrays must be initiated with data type int, otherwise pandas would auto convert to float
    df_cycles = pd.DataFrame(
        {
            'end_index': np.array([], dtype=int),
            'event_time_aligned': [],
            'event_time_real': [],
            'event_length': [],
            'event_phase': [],
            'event_lin_phase': [],
            'event_count': [],
            'raw': [],
            'filtered': [],
            'envelope': [],
            'offline_phase': [],
            'estimated_phase': []
        }
    )

    if event_idx is not None and phase is not None:
        stim_phase = phase[event_idx]
        stim_lin_phase = lin_phase[event_idx]
    elif event_time is not None and phase is not None:
        event_idx = closest_indices(event_time, time)
        stim_phase = phase[event_idx]
        stim_lin_phase = lin_phase[event_idx]
    else:
        stim_phase = None
        stim_lin_phase = None

    for i, t in enumerate(end_index):
        try:

            if event_time is not None:
                # real event time
                event_incl = np.logical_and((event_timestamp >= timestamp[t]),
                                            (event_timestamp < timestamp[end_index[i + 1]]))
                event_time_real = event_time[event_incl]
                # relative event time in milliseconds aligned to cycle start
                event_time_aligned = (event_time_real - time[t]) * 1000
                event_count = len(event_time_aligned)
                # event phase
                event_phase = stim_phase[event_incl]
                event_lin_phase = stim_lin_phase[event_incl]
                if event_length is not None:
                    # event duration
                    # event_duration = event_length[event_incl]
                    pass
                else:
                    # event_duration = None
                    pass
            else:
                event_time_real = None
                event_time_aligned = None
                event_phase = None
                event_lin_phase = None
                # event_duration = None
                event_count = 0

            if estimated_phase is not None:
                estm_phase = estimated_phase[t:end_index[i + 1]]
            else:
                estm_phase = None

            df_cycles = pd.concat([
                df_cycles,
                pd.DataFrame(
                    {
                        'end_index': [t],
                        'event_time_aligned': [event_time_aligned],
                        'event_time_real': [event_time_real],
                        'event_phase': [event_phase],
                        'event_lin_phase': [event_lin_phase],
                        'event_count': [event_count],
                        'raw': [raw[t:end_index[i + 1]]],
                        'filtered': [filtered[t:end_index[i + 1]]],
                        'envelope': [envelope[t:end_index[i + 1]]],
                        'offline_phase': [phase[t:end_index[i + 1]]],
                        'estimated_phase': [estm_phase]
                    }
                )
            ])

        except IndexError:
            pass
            
    if label_in_reward:
        # label each cycle as in or out of reward period
        reward_period = get_sensor_on_period(SessionData, reward_sensor_name)
        cycle_in_reward = np.isin(timestamp[df_cycles['end_index'].to_numpy()], reward_period)
        df_cycles['in_reward'] = cycle_in_reward.tolist()

    if label_after_first_event:
        # label each cycle as before or after the first event
        df_cycles['after_first_event'] = timestamp[df_cycles['end_index'].to_numpy()] > event_timestamp[0]

    if label_above_threshold:
        # label each cycle as above amplitude threshold or not
        df_cycles['avg_amplitude'] = df_cycles['envelope'].apply(np.mean, axis=0)
        df_cycles['above_threshold'] = df_cycles['avg_amplitude'] >= amplitude_threshold

    return df_cycles


def cycles_info(df_cycles, label):
    print('-------------' + label + ' info-------------')
    event_count_arr = np.array(df_cycles.get('event_count'))
    print(f'Total # of cycles:{len(event_count_arr)}')
    print(f'# of multiple-event cycles:{sum(event_count_arr > 1)}')
    print(f'# of skipped cycles:{sum(event_count_arr == 0)}')
    print('--------------------------------------------')
    

def count_phases_in_range(
    phase_array, 
    target_range_center, 
    target_range_width=np.pi/2
    ):
    # Normalize phase values to the range [0, 2*pi)
    normalized_phases = np.mod(phase_array, 2 * np.pi)

    # Calculate the lower and upper bounds of the target range
    lower_bound = (target_range_center - target_range_width/2) % (2 * np.pi)
    upper_bound = (target_range_center + target_range_width/2) % (2 * np.pi)

    # Count the number of phases within the target range
    if lower_bound > upper_bound:
        # target range includes 0 / 2pi
        count = np.sum((normalized_phases >= lower_bound) | (normalized_phases <= upper_bound))
    else:
        count = np.sum((normalized_phases >= lower_bound) & (normalized_phases <= upper_bound))

    return count