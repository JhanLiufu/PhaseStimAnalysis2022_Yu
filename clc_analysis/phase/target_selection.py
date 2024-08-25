import numpy as np
from scipy import signal


def index_in(SessionData, num_std=0):
    """
    returns an array of the index of data where signal amplitude in the frequency range of
    interest exceeds certain bar

    Parameters
    ----------
    SessionData : SessionData class object which contains data of a session
    num_std : int, number of standard deviation above average signal magnitude to use as magnitude threshold

    Returns
    -------
    index_in : array of index
    """
    filtered_envelope = np.abs(SessionData.filtered)
    bar = np.mean(filtered_envelope) + num_std*np.std(filtered_envelope)
    index_in = np.array(filtered_envelope) > bar
    return index_in


def select_target(SessionData, target, index_in):
    """
    get the index of target phases in raw LFP/time/timestamp array

    Parameters
    ----------
    SessionData : SessionData class object which contains data of a session
    target : float, phase target value
    index_in : an array of the index of data where signal amplitude in the frequency range of
    interest exceeds certain bar

    Returns
    -------
    array of index
    """
    # PROBLEM: need to use index_in together with peaks index array
    if target == np.pi:

        target_index, _ = signal.find_peaks(SessionData.filtered, height=0)
        target_time = SessionData.time[target_index]

    elif target == 0:

        target_index, _ = signal.find_peaks(-SessionData.filtered, height=0)
        target_time = SessionData.time[target_index]

    else:

        # calculate phase
        phase = np.angle(signal.hilbert(SessionData.filtered)) + np.pi
        target_index = []

        # TESTED
        if target < np.pi:
            # if target in the rising phase, locate troughs first
            pivots, _ = signal.find_peaks(-SessionData.filtered, height=0)

            for p in pivots:
                curr_pivot = p

                try:
                    while phase[curr_pivot] < target or phase[curr_pivot] > np.pi:
                        curr_pivot += 1
                except IndexError:
                    continue

                target_index.append(curr_pivot)

        # TESTED
        elif target > np.pi:
            # if target in the falling phase, locate peaks first
            pivots, _ = signal.find_peaks(SessionData.filtered, height=0)

            for p in pivots:
                curr_pivot = p

                try:
                    while phase[curr_pivot] < target:
                        curr_pivot += 1
                except IndexError:
                    continue

                target_index.append(curr_pivot)

        target_index = np.array(target_index)
        target_time = SessionData.time[target_index]

    return target_index, target_time