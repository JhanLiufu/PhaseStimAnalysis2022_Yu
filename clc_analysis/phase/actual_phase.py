import numpy as np


def actual_phase(phase_ref: object, event_time: object, time: object) -> object:
    """
    Find the phases of oscillatory signal at the time when events happened

    Parameters
    ----------
    phase_ref : array of phase value of oscillatory signal
    event_time : array of event time points in seconds
    time : array of time values in seconds, should have the same length as phase_ref

    Returns
    -------
    array of event phases corresponding to event_time
    """
    actual_phase = []
    start = 0

    for stim in event_time:
        for i in range(start, len(time)):
            try:
                if time[i] <= stim <= time[i + 1]:
                    actual_phase.append(phase_ref[i])
                    start = i
                    break
            except IndexError:
                pass

    return np.array(actual_phase)