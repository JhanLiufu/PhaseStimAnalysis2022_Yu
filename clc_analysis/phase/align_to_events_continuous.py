import numpy as np


def align_to_events_continuous(data_in, times_in, ref_events, win):
    """
    Aligns a continuous data to reference events.
    Example usage: align LFP relative to known events.
    All event times are in seconds

    :param data_in: array of values (LFP signal)
    :param times_in: array of timestamps for values
    :param ref_events: array of event times to align to
    :param win: window length around alignment time in seconds
    :return: array of event values and times aligned to ref_events
    """

    arr_out_data = []
    arr_out_time = []

    for e in ref_events:

        # get time indices around window
        try:
            ind_start = np.argmax(times_in >= e - win)
            ind_end = np.argmax(times_in >= e + win)
            data_incl = data_in[ind_start:ind_end]
            time_incl = times_in[ind_start:ind_end] - e
            arr_out_data.append(data_incl)
            arr_out_time.append(time_incl)
        except:
            print(f'{e} failed')

    return arr_out_data, arr_out_time
