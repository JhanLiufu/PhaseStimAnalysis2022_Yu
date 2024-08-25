"""
phase_plot: methods to generate the primary analysis plots for CLC phase detection
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from collections import OrderedDict
from sklearn.linear_model import LinearRegression as lin_reg


def stim_len_hist(SessionData, SessionParam, pump_name, upper_bound=30, hist_width=0.25, y_axis='count', dir_fig=None):
    """
    Make a histogram of pump event duration

    Parameters
    ----------
    SessionData : SessionData class object which contains data of a session
    SessionParam : SessionParam class object, which contains session parameter info
    pump_name : string, DIO pump name
    upper_bound : float, upper bound of the histogram. default value is 30
    hist_width : float, bin size/width of the histogram. default value is 0.25
    y_axis : string, 'Count'/'count' OR 'prob'/'Prob'. The y-axis of the histogram can be either probability
    or event count
    dir_fig: string, full path for savefig
    """
    plt.figure()

    stim_len = SessionData.pump_data.get(pump_name).get('length')
    len_hist, len_edges = np.histogram(stim_len, np.arange(0, upper_bound, hist_width))

    if y_axis == 'count' or y_axis == 'Count':
        plt.bar(len_edges[:-1], len_hist, width=hist_width)
        plt.ylabel('Event count')

    elif y_axis == 'prob' or y_axis == 'Prob':
        plt.plot(len_edges[:-1], len_hist/sum(len_hist))
        plt.ylabel('Probability')

    else:
        print('y axis is either probability or event count; y_axis = count or prob')

    plt.xlabel('Time (ms)')
    plt.title(f'{SessionParam.date} S{SessionParam.session} Event Length Dist. (n={len(stim_len)})')

    if dir_fig is not None:
        full_path = dir_fig + '/stim_len_hist.pdf'
        plt.savefig(full_path)


def event_phase_hist(phase_lst, label_lst, color_lst=None, target_lst=None, width=0.1, norm=False, dir_fig=None, plot=True, session_param=None, close_fig=False):
    """
    Make a polar histogram (or rose plot) of event phases

    Parameters
    ----------
    phase_lst : list of circular phase arrays
    label_lst : list of strings, labels that correspond to each df_cycle
    color_lst: list of strings, color names
    target_lst: list of target phase values (0~2pi)
    plot: boolean, whether to show a plot or not
    session_param : SessionParam class object, which contains session parameter info
    norm: boolean, normalize histogram or not, default to False
    width : float, bin size of the histogram. default value set to 0.1
    dir_fig: string, full path for savefig
    close_fig: False
    """
    if plot:
        plt.figure()
        ax_polar = plt.subplot(projection='polar')
        ax_polar.set_yticklabels([])

    sample_count_str = ''
    dict_event_phases = {}

    for i, (phases, label) in enumerate(zip(phase_lst, label_lst)):
        # phase_hist, phase_edges = np.histogram(phases, np.arange(0, 2 * np.pi, width))
        phase_hist, phase_edges = np.histogram(phases, np.arange(-np.pi, np.pi, width))
        if norm:
            phase_hist = phase_hist / sum(phase_hist)
        dict_event_phases.update({label: {'phases': phases, 'hist': phase_hist, 'edges': phase_edges}})

        if plot:
            if color_lst is None:
                ax_polar.plot(phase_edges, np.append(phase_hist, phase_hist[0]), label=label)
            else:
                ax_polar.plot(phase_edges, np.append(phase_hist, phase_hist[0]), label=label, c=color_lst[i])
                
            if target_lst is not None:
                if color_lst is None:
                    ax_polar.plot([target_lst[i], target_lst[i]], [0, 1], linestyle='--')
                else:
                    ax_polar.plot([target_lst[i], target_lst[i]], [0, 1], linestyle='--', c=color_lst[i])

        xL = [
            r'$-\pi$', r'$-\frac{\pi}{4}$', r'$-\frac{\pi}{2}$', r'$-\frac{3\pi}{4}$',
            '0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$'
        ] 
        ax_polar.set_xticks(np.pi/180. * np.linspace(-180, 180, 8, endpoint=False))
        ax_polar.set_xticklabels(xL)
        ax_polar.set_thetalim(-np.pi, np.pi)

        # add sample count for each group
        event_count = len(phases)
        sample_count_str += f'; n{i+1}={event_count}'

    if plot:
        if session_param is not None:
            plt.title(f'{session_param.session_name} event phase dist.'+sample_count_str)
        else:
            plt.title('Event phase dist.'+sample_count_str)
        plt.legend()

        if dir_fig is not None:
            full_path = dir_fig + '/event_phase_hist.pdf'
            plt.savefig(full_path)

        if close_fig:
            plt.close()

    return dict_event_phases


def event_aligned_time_hist(df_cycle_lst, label_lst, SessionParam, width=2, upper_bound=140, dir_fig=None):
    """
    Make a histogram of event time aligned to the start of oscillatory cycles

    Parameters
    ----------
    df_cycle_lst : list of dataframes, each contains data group by cycles. refer to organize_cycle.organize_cycle
    label_lst : list of strings, labels that correspond to each df_cycle
    SessionParam : SessionParam class object, which contains session parameter info
    width : float, bin size of the histogram. default value set to 2
    upper_bound : float, upper bound of the histogram. default value set to 140
    dir_fig: string, full path for savefig
    """
    plt.figure()

    sample_count_str = ''

    for i, (df_cycles, label) in enumerate(zip(df_cycle_lst, label_lst)):
        time_hist, time_edges = np.histogram(np.hstack(df_cycles.get('event_time_aligned')),
                                             np.arange(0, upper_bound, width))

        # plot curve for each df_cycle
        plt.plot(time_edges[:-1],
                time_hist / sum(time_hist),
                label=label)

        # add sample count for each group
        event_count = len(np.hstack(df_cycles.get('event_time_real')))
        sample_count_str += f'; n{i+1}={event_count}'

    plt.title(f'{SessionParam.session_name} stim time dist.' + sample_count_str)
    plt.xlabel('Time from cycle start')
    plt.ylabel('Probability')
    plt.legend()

    if dir_fig is not None:
        full_path = dir_fig + '/event_aligned_time_hist.pdf'
        plt.savefig(full_path)


def event_lag_xcorr_hist(df_cycle_lst, label_lst, SessionParam, width=2, upper_bound=100, dir_fig=None):
    """
    Make a cross correlogram of event time points and target time points

    Parameters
    ----------
    df_cycle_lst : list of dataframes, each contains data group by cycles. refer to organize_cycle.organize_cycle
    label_lst : list of strings, labels that correspond to each df_cycle
    SessionParam : SessionParam class object, which contains session parameter info
    width : float, bin size of the histogram. default value set to 2
    upper_bound : float, upper bound of the histogram. default value set to 100
    dir_fig: string, full path for savefig
    """
    sample_count_str = ''

    for i, (df_cycles, label) in enumerate(zip(df_cycle_lst, label_lst)):

        target_time = np.hstack(df_cycles.get('target_time_real'))
        event_time = np.hstack(df_cycles.get('event_time_real'))
        lag = np.hstack([e - target_time for e in event_time])*1000

        lag_hist, lag_edges = np.histogram(lag, np.arange(0, upper_bound, width))
        plt.plot(lag_edges[:-1],
                 lag_hist/sum(lag_hist),
                 label=label)

        event_count = len(event_time)
        sample_count_str += f'; n{i+1}={event_count}'

    plt.title(f'{SessionParam.date} S{SessionParam.session} lag xcorr dist.' + sample_count_str)
    plt.xlabel('Lag (ms)')
    plt.ylabel('Probability')
    plt.legend()

    if dir_fig is not None:
        full_path = dir_fig + '/event_lag_xcorr_hist.pdf'
        plt.savefig(full_path)


def get_diff_dist(a1, a2):
    """
    Get a distrbution of time differences (in ms) between two time series

    Parameters
    ----------
    a1 : list or array, (lagging) time series to subtract from
    a2 : list or array, (leading) time series to subtract with
    """
    dist = []
    for e in a1:
        try:
            curr = a2[a2 < e][-1]
            dist.append(e - curr)
        except IndexError:
            pass

    return np.array(dist)


def quarter_period_hist(SessionData, SessionParam, width=2, upperbound=150, dir_fig=None):
    """
    Make a histogram of quarter periods (dividing each cycle into four sections)

    Parameters
    ----------
    SessionData: SessionData class object, which contains session data
    SessionParam : SessionParam class object, which contains session parameter info
    width : float, bin size of the histogram. default value set to 2
    upperbound : float, upper bound of the histogram. default value set to 150
    dir_fig: string, full path for savefig
    """
    # locate the four markpoints
    sign_change = np.diff(np.sign(SessionData.filtered))
    falling = np.where(sign_change == -2)[0]
    rising = np.where(sign_change == 2)[0]
    peaks, _ = find_peaks(SessionData.filtered)
    troughs, _ = find_peaks(SessionData.filtered)

    hist_lst = []
    edges_lst = []
    label_lst = ['falling2trough', 'peak2falling', 'rising2peak', 'trough2rising']

    markpoints = [troughs, falling, peaks, rising, troughs]
    for i, e in enumerate(markpoints):
        try:
            hist, edges = np.histogram(get_diff_dist(SessionData.time[e] * 1000,
                                                     SessionData.time[markpoints[i + 1]] * 1000),
                                       np.arange(0, upperbound, width))
            hist_lst.append(hist)
            edges_lst.append(edges)
        except IndexError:
            pass

    for h, e, l in zip(hist_lst, edges_lst, label_lst):
        plt.plot(e[:-1], h / sum(h), label=l)

    plt.ylabel('Probability')
    plt.xlabel('Time (ms)')
    plt.title(f'{SessionParam.date} S{SessionParam.session} quarter period dist.')
    plt.legend()

    if dir_fig is not None:
        full_path = dir_fig + '/quarter_period_hist.pdf'
        plt.savefig(full_path)


def estm_phase_error_dist(phase_lst, label_lst, offline_phase, width=0.1, dir_fig=None, plot=True, session_param=None):
    """

    Parameters
    ----------
    phase_lst: array like, list of arrays of estimated phases
    label_lst: array like, list of labels
    offline_phase: array like, array of offline / ground truth phases
    width: float, bin size of histogram
    dir_fig: string, full path for savefig
    plot: boolean, whether to show a plot
    session_param:

    Returns
    -------

    """
    edges = np.arange(0, 2*np.pi, width)
    # signal_mask = SessionData.envelope/1000 >= threshold
    dict_vals = {}

    if plot:
        plt.figure()
        ax_polar = plt.subplot(projection='polar')
        ax_polar.set_yticklabels([])

    for p, l in zip(phase_lst, label_lst):
        means = []
        stds = []
        # phase bin index each sample belongs to
        bin_idx = np.digitize(offline_phase, edges) - 1
        # each field is a list of phase error in that phase interval
        vals = OrderedDict()
        for b in bin_idx:
            vals[b] = []

        df_err = pd.DataFrame({  # 'true_phase': SessionData.phase[signal_mask],
                               'true_phase': offline_phase,
                                 # 'inst_phase': np.append([None]*(len(SessionData.phase) - len(p)), p)[signal_mask],
                               'inst_phase': p,
                               'bin_idx': bin_idx})

        for ind, row in df_err.iterrows():
            try:
                vals[row['bin_idx']].append(row['inst_phase'] - row['true_phase'])
            except TypeError:
                pass

        # take mean and std of each bin
        for v in vals.values():
            means.append(np.mean(v))
            stds.append(np.std(v))

        dict_vals.update({l: {'err_binned': vals,
                              'means': means,
                              'stds': stds,
                              'edges': edges}})

        if plot:
            ax_polar.plot(edges, means, label=l)
            # plt.fill_between(edges, np.subtract(means, stds), np.add(means, stds), facecolor=c, alpha=0.4)
            # plt.axhline(y=0, linestyle='--', c='r')
            # plt.xlabel('True Phase (rad)')
            # plt.ylabel('Estm. Phase err (rad)')

    if plot:
        if session_param is not None:
            plt.title(f'{session_param.session_name} Estimation error against Phase')
        else:
            plt.title(f'Estimation error against Phase')

        if dir_fig is not None:
            full_path = dir_fig + '/estm_phase_error_dist.pdf'
            plt.savefig(full_path)

    return dict_vals


def estm_phase_against_phase(SessionData, SessionParam, phase_history, label, dot_size=0.005, fltr_buffer_size=400,
                             dir_fig=None):
    plt.ylim([0, 2 * np.pi])
    plt.scatter(SessionData.phase[fltr_buffer_size:-(len(SessionData.phase)-len(phase_history)-fltr_buffer_size)],
                phase_history, s=dot_size)

    plt.xlabel('True Phase (rad)')
    plt.ylabel('Estm. Phase (rad)')
    plt.title(f'{SessionParam.date} Session {SessionParam.session} {label} estm. phase vs. true phase')

    if dir_fig is not None:
        full_path = dir_fig + '/estm_phase_against_phase_hist.pdf'
        plt.savefig(full_path)


def generate_random_phase(len, dispersion=0):
    """
    Generate an array of random phase value between 0 and 2pi

    Parameters
    ----------
    len : length of return array

    Returns
    -------
    rand_phase: array of random phase values
    """
    rand_phase = np.random.vonmises(np.pi, dispersion, len)
    rand_phase += 2 * np.pi * (rand_phase < 0)

    # assume each event/phase corresponds to a separate cycle
    rand_phase_lin = []
    for i, p in enumerate(rand_phase):
        rand_phase_lin.append(p + i * 2 * np.pi)

    return {'phases': rand_phase, 'lin_phases': rand_phase_lin}


def df_cycles_to_lin_phase(df_cycles):
    """
    Generate array of linear event phases from df_cycles dataframe

    Parameters
    ----------
    df_cycles: pandas dataframe, see organize_cycles()

    Returns
    -------
    array of linear event phases
    """
    lin_phase = np.array([])
    # turn event phases into linearly growing values
    for ind, row in df_cycles.iterrows():
        lin_phase = np.append(lin_phase, np.array(row['event_phase']) + (ind * 2 * np.pi))

    return lin_phase


def event_phase_self_xcorr(lin_phase_lst, label_lst, color_lst=None, lower_bound=np.pi/2,
                           upper_bound=50*np.pi/2, width=0.4,
                           dir_fig=None, plot=True, plot_random=True, session_param=None, close_fig=False):
    """
    Generate self-correlogram of event phases, used to visualize global periodicity
    of events. Perfectly periodic event series should have distinct peaks at integer
    multiples of 2pi. Plot self-correlogram of randomly generated phases for comparison.

    Parameters
    ----------
    lin_phase_lst : list of dataframes, each contains data group by cycles. refer to organize_cycle.organize_cycle
    label_lst : list of strings, labels that correspond to each df_cycle
    color_lst: list of strings, color names
    session_param : SessionParam class object, which contains session parameter info
    lower_bound: lower bound of histogram
    upper_bound: upper bound of histogram
    width : float, bin size of the histogram. default value set to 0.4
    dir_fig: string, full path for savefig
    plot: bool, whether to make the plo
    plot_random: whether to plot a sequence of completely random phases for comparison
    close_fig: whether to close figure

    Returns
    -------

    """
    if plot:
        plt.figure()

    dict_hist_edges = {}
    
    for i, (lin_phase, label) in enumerate(zip(lin_phase_lst, label_lst)):

        phase_self_xcorr = np.hstack([lin_phase - p for p in lin_phase])
        phase_self_hist, phase_self_edges = np.histogram(phase_self_xcorr, np.arange(lower_bound, upper_bound, width))

        dict_hist_edges.update({label: {'hist': phase_self_hist, 'edges': phase_self_edges}})

        if plot:
            if color_lst is None:
                plt.plot(phase_self_edges[:-1],
                        phase_self_hist / sum(phase_self_hist),
                        label=label)
            else:
                plt.plot(phase_self_edges[:-1],
                        phase_self_hist / sum(phase_self_hist),
                        label=label, c=color_lst[i])

    # plot random control
    if plot:
        if plot_random:
            rand_phase = generate_random_phase(len(lin_phase_lst[0])).get('lin_phases')
            rand_self_xcorr = np.hstack([rand_phase - r for r in rand_phase])
            rand_self_hist, rand_self_edges = np.histogram(rand_self_xcorr, np.arange(lower_bound, upper_bound, width))

            plt.plot(rand_self_edges[:-1],
                     rand_self_hist / sum(rand_self_hist),
                     label='rand_vonmises')

        if session_param is not None:
            plt.title(f'{session_param.session_name} Event Phase Self Correlogram')
        else:
            plt.title('Event Phase Self Correlogram')

        plt.xlabel('Phase')
        plt.ylabel('Probability')
        plt.legend()

        if dir_fig is not None:
            full_path = dir_fig + '/event_phase_self_xcorr.pdf'
            plt.savefig(full_path)

        if close_fig:
            plt.close()

    return dict_hist_edges


def average_coherence(e, h, cycles=20, win=np.pi / 8):
    coherence_arr = []
    for i in range(1, cycles + 1):
        mask = np.abs(e - i * 2 * np.pi) <= win
        coherence_arr = np.concatenate([coherence_arr, h[mask]])

    return coherence_arr, np.mean(coherence_arr)


def average_contrast(phase_xcorr_hist):
    peaks_x, _ = find_peaks(phase_xcorr_hist)
    troughs_x, _ = find_peaks(-phase_xcorr_hist)

    peaks = phase_xcorr_hist[peaks_x]
    troughs = phase_xcorr_hist[troughs_x]

    contrast = np.hstack([peaks - t for t in troughs])
    contrast_avg = np.mean(contrast)

    return contrast_avg


def coherence_contrast_decay(phase_xcorr_hist_lst, phase_xcorr_edges_lst, label_lst,
                             threshold=None, color_lst=None,
                             plot=True, session_param=None, dir_fig=None, close_fig=False):
    """

    Parameters
    ----------
    phase_xcorr_hist_lst
    phase_xcorr_edges_lst
    label_lst
    threshold
    color_lst
    plot
    session_param
    dir_fig
    close_fig

    Returns
    -------

    """
    if plot:
        plt.figure()

    dict_coh = {}

    for i, (phase_xcorr_hist, phase_xcorr_edges, label) in enumerate(zip(phase_xcorr_hist_lst,
                                                                         phase_xcorr_edges_lst,
                                                                         label_lst)):
        coherence_decay = []
        e_x = []
        phase_xcorr_hist_norm = phase_xcorr_hist / sum(phase_xcorr_hist)
        for j in np.arange(np.argmax(phase_xcorr_edges > 4 * np.pi), len(phase_xcorr_edges), 10):
            e_x.append(phase_xcorr_edges[j] / (2 * np.pi))
            coherence_decay.append(average_contrast(phase_xcorr_hist_norm[:j]))

        dict_coh.update({label: {'coherence': coherence_decay, 'cycle_num': e_x}})

        if plot:
            if color_lst is None:
                plt.plot(np.array(e_x), coherence_decay, label=label)
            else:
                plt.plot(np.array(e_x), coherence_decay, label=label, c=color_lst[i])

            if threshold is not None:
                # plot coherence threshold as a horizontal line
                if color_lst is None:
                    plt.axhline(y=threshold, linestyle='--', c='r')
                else:
                    plt.axhline(y=threshold, linestyle='--', c=color_lst[i])

    if plot:
        plt.ylabel('Coherence contrast')
        plt.xlabel('Number of cycles')
        plt.legend()

        if session_param is not None:
            plt.title(f'{session_param.session_name} Coherence decay curve')
        else:
            plt.title('Coherence decay curve')

        if dir_fig is not None:
            full_path = dir_fig + '/Coherence decay curve.pdf'
            plt.savefig(full_path)

        if close_fig:
            plt.close()

    return dict_coh


def neighbor_phase_diff_hist(lin_phase_lst, label_lst, color_lst=None, upper_bound=4*np.pi, width=0.05, dir_fig=None,
                             plot=True, session_param=None, close_fig=False):
    """
    Generate histogram of neighbor phase differences, used to visualize local periodicity
    of events. Perfectly periodic event series should have a distinct peak at integer 2pi

    Parameters
    ----------
    lin_phase_lst : list of dataframes, each contains data group by cycles. refer to organize_cycle.organize_cycle
    label_lst : list of strings, labels that correspond to each df_cycle
    color_lst: list of strings, color names
    session_param : SessionParam class object, which contains session parameter info
    upper_bound: upper bound of histogram. default value set to 4pi
    width : float, bin size of the histogram. default value set to 0.05
    dir_fig: string, full path for savefig
    plot: boolean, whether to plot the histogram
    close_fig: whether to close the figure generated

    Returns
    -------
    dict_neighbor_diff
    """
    if plot:
        plt.figure()

    sample_count_str = ''
    dict_neighbor_diff = {}

    for i, (lin_phase, label) in enumerate(zip(lin_phase_lst, label_lst)):

        neighbor_diff = np.diff(lin_phase)

        sample_count = len(neighbor_diff)
        sample_count_str += f'; n{i + 1}={sample_count}'

        hist, edges = np.histogram(neighbor_diff, np.arange(0, upper_bound, width))
        dict_neighbor_diff.update({label: {'neighbor_diff': neighbor_diff, 'hist': hist, 'edges': edges}})

        if plot:
            if color_lst is None:
                plt.plot(edges[:-1], hist / sum(hist), label=label)
            else:
                plt.plot(edges[:-1], hist / sum(hist), label=label, c=color_lst[i])

    if plot:
        plt.xlabel('Neighbor Phase Difference')
        plt.ylabel('Probability')
        plt.legend()
        
        if session_param is not None:
            plt.title(f'{session_param.session_name} neighbor phase diff. hist. ' + sample_count_str)
        else:
            plt.title('Neighbor phase diff. hist.')

        if dir_fig is not None:
            full_path = dir_fig + '/neighbor_phase_diff_hist.pdf'
            plt.savefig(full_path)

        if close_fig:
            plt.close()

    return dict_neighbor_diff


def calculate_npvi(p_diff):
    npvi = []
    for i, v in enumerate(p_diff):
        try:
            npvi.append(abs((v - p_diff[i - 1]) / ((v + p_diff[i - 1]) / 2)))
        except IndexError:
            pass

    return np.mean(npvi)


def rising_falling_lin_reg(filtered_signal, dir_fig=None, plot=True, session_param=None, x_lim=200, fs=1500,
                           annotate=True, close_fig=False):
    """
    Scatter plot and linear regression between rising and falling half cycle lengths; cycle asymmetry
    lenghts in unit of milliseconds

    Parameters
    ----------
    filtered_signal: array-like, filtered LFP signal
    dir_fig: string, full path for savefig
    plot: boolean, generate plot or not
    session_param: class object, information of a session
    x_lim: maximum length to plot
    fs
    annotate: boolean, write down the fitted equation or not
    close_fig

    Returns
    -------
    dict_cyc_asym: dictionary that contains array of rising cycle lengths,
    array of falling cycle lengths (in samples), regression slope and intercept
    """
    # Indentify peaks and troughs
    p_idx, _ = find_peaks(filtered_signal)
    t_idx, _ = find_peaks(-filtered_signal)

    # Calculate rising and falling cycle lengths
    rising_lens = []
    falling_lens = []

    for i in range(len(p_idx)):
        try:
            falling_lens.append(t_idx[i] - p_idx[i])
            rising_lens.append(p_idx[i + 1] - t_idx[i])
        except IndexError:
            pass

    len_diff = len(rising_lens) - len(falling_lens)
    # print(len_diff)
    if len_diff > 0:
        rising_lens = rising_lens[:-len_diff]
    elif len_diff < 0:
        falling_lens = falling_lens[:len_diff]

    # convert unit to milliseconds
    rising_lens = (np.array(rising_lens) / fs) * 1000
    falling_lens = (np.array(falling_lens) / fs) * 1000

    falling_lens_fit = np.array(falling_lens).reshape((-1, 1))
    model = lin_reg().fit(falling_lens_fit, rising_lens)

    # print(f"intercept: {model.intercept_}")
    # print(f"slope: {model.coef_}")

    if plot:
        plt.figure()
        plt.xlim([0, x_lim])
        plt.ylim([0, x_lim])
        plt.scatter(falling_lens, rising_lens, s=0.3)
        fit_x = np.arange(0, x_lim, 1)
        # plot the fitted line
        plt.plot(fit_x, model.coef_[0] * fit_x + model.intercept_)

        plt.xlabel('Falling cycle length (ms)')
        plt.ylabel('Rising cycle length (ms)')

        if session_param is not None:
            plt.title(f'{session_param.session_name} rising falling lin reg ')
        else:
            plt.title('Rising falling lin reg')

        if dir_fig is not None:
            full_path = dir_fig + '/rising_falling_lin_reg.pdf'
            plt.savefig(full_path)

        if close_fig:
            plt.close()

    return {'rising_lens': rising_lens, 'falling_lens': falling_lens_fit,
            'intercept': model.intercept_, 'slope': model.coef_[0]}


def power_distribution(envelope_lst, label_lst, color_lst=None, upper_bound=300, width=5,
                       plot=True, dir_fig=None, session_param=None, close_fig=False):
    """

    Parameters
    ----------
    envelope_lst
    label_lst
    upper_bound
    width
    plot
    dir_fig
    session_param
    close_fig: whether to close the figure after plotting

    Returns
    -------

    """
    if plot:
        plt.figure()

    dict_power_dist = {}

    for i, (env, label) in enumerate(zip(envelope_lst, label_lst)):
        hist, edges = np.histogram(env, np.arange(0, upper_bound, width))
        hist = hist / sum(hist)
        dict_power_dist.update({label: {'hist': hist,
                                        'edges': edges,
                                        'mean': np.mean(env),
                                        'var': np.var(env)}})

        if plot:
            if color_lst is None:
                plt.plot(edges[:-1], hist, label=label)
            else: 
                plt.plot(edges[:-1], hist, label=label, c=color_lst[i])
                
    if plot:
        plt.xlabel('Power')
        plt.ylabel('Probability')
        plt.legend()
        
        if session_param is not None:
            plt.title(f'{session_param.session_name} power distribution')
        else:
            plt.title('Power distribution')

        if dir_fig is not None:
            full_path = dir_fig + '/power_dist.pdf'
            plt.savefig(full_path)

        if close_fig:
            plt.close()

    return dict_power_dist






