import numpy as np
from scipy.signal import hilbert, sosfiltfilt, butter, freqz
import random


def echt(xr, b, a, Fs, n=None):
    """
    endpoint-correcting hilbert transform

    Parameters
    ----------
    xr: array like, input signal
    b: numerators of IIR filter response
    a: denominator of IIR filter response
    Fs: signal sampling rate
    n

    Returns
    -------
    analytic signal

    """
    # Check input
    if n is None:
        # default: n is length of xr
        n = len(xr)
    if not all(np.isreal(xr)):
        xr = np.real(xr)

    # Compute FFT
    x = np.fft.fft(xr, n)

    # Set negative components to zero and multiply positive by 2 (apart from DC and Nyquist frequency)
    h = np.zeros(n, dtype=x.dtype)
    if n > 0 and 2 * (n // 2) == n:
        # even and non-empty
        h[[0, n // 2]] = 1
        h[1:n // 2] = 2
    elif n > 0:
        # odd and non-empty
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    x = x * h

    # Compute filter's frequency response
    # filt_order = 2
    # b, a = butter(filt_order, np.array([filt_lowcut, filt_hightcut]) / (Fs / 2), btype='bandpass')
    T = 1 / Fs * n
    # filt_freq = (np.arange(-n // 2, n // 2) / T)
    filt_freq = np.ceil(np.arange(-n/2, n/2)) / T
    filt_coeff = freqz(b, a, worN=filt_freq, fs=Fs)

    # Multiply FFT by filter's response function
    x = np.fft.fftshift(x)
    # x = x * filt_coeff
    x = x * filt_coeff[1]
    x = np.fft.ifftshift(x)

    # IFFT
    x = np.fft.ifft(x)

    # return (np.squeeze(x))[0]
    return x


def endpoint_correcting_hilbert_transform(signal, target_phase, filter_lowcut=6, filter_highcut=9,
                                          fs_filter=1500, fltr_buffer_size=400, fltr_order=2, stim_ok_reset=None):
    """
    Phase detection with endpoint-correcting Hilbert transform
    based on Schrelgmann etal 2021

    Parameters
    ----------
    signal
    target_phase
    filter_lowcut
    filter_highcut
    fs_filter
    fltr_buffer_size
    fltr_order
    stim_ok_reset: where to reset stim_ok to True

    Returns
    -------

    """
    event_index = []
    phase_history = [-1]*fltr_buffer_size
    stim_ok = True
    prev_phase = -1

    b, a = butter(fltr_order, np.array([filter_lowcut, filter_highcut]) / (fs_filter / 2), btype='bandpass')

    # print(signal[:20])
    # raise Exception('breakpoint')

    for i in range(fltr_buffer_size, len(signal)):
        curr_buffer = signal[i - fltr_buffer_size:i]

        curr_analytic = echt(curr_buffer, b, a, fs_filter)
        curr_phase = np.angle(curr_analytic)[-1]
        phase_history.append(curr_phase)

        # at trough, phase jumps from 2pi to 0
        # phase should be strictly increasing everywhere else
        if (prev_phase - curr_phase) > np.pi and stim_ok_reset is None:
            stim_ok = True

        elif stim_ok_reset is not None:
            if curr_phase >= stim_ok_reset and stim_ok is False:
                stim_ok = True

        prev_phase = curr_phase

        if curr_phase >= target_phase and stim_ok:
            event_index.append(i)
            stim_ok = False

    return np.array(event_index), np.array(phase_history)


def ecHT_random_target(
    signal, 
    filter_lowcut=6, 
    filter_highcut=9,
    fs_filter=1500, 
    fltr_buffer_size=400, 
    fltr_order=2
):
    
    event_index = []
    phase_history = [-1]*fltr_buffer_size
    stim_ok = True
    prev_phase = -1
    target_phase = np.pi
    
    b, a = butter(fltr_order, np.array([filter_lowcut, filter_highcut]) / (fs_filter / 2), btype='bandpass')

    for i in range(fltr_buffer_size, len(signal)):
        # Crop out current window
        curr_buffer = signal[i - fltr_buffer_size : i]

        # Compute phase at the endpoint
        curr_analytic = echt(curr_buffer, b, a, fs_filter)
        curr_phase = np.angle(curr_analytic)[-1]
        phase_history.append(curr_phase)

        # Reset cycle at trough, phase jumps from 2pi to 0
        # phase should be strictly increasing everywhere else
        if (prev_phase - curr_phase) > np.pi:
            stim_ok = True
            # randomly change target
            target_phase = np.random.vonmises(0, 0) # + np.pi

        prev_phase = curr_phase

        if curr_phase >= target_phase and stim_ok:
            event_index.append(i)
            stim_ok = False

    return np.array(event_index), np.array(phase_history)


def ecHT_random_delay(
    signal, 
    filter_lowcut=6, 
    filter_highcut=9,
    fs_filter=1500, 
    fltr_buffer_size=400, 
    fltr_order=2
):
    # Maximum delay - period of the lowest frequency
    max_delay = int((1/filter_lowcut)*fs_filter)
    
    event_index = []
    phase_history = [-1]*fltr_buffer_size
    prev_phase = -1
    
    b, a = butter(fltr_order, np.array([filter_lowcut, filter_highcut]) / (fs_filter / 2), btype='bandpass')

    for i in range(fltr_buffer_size, len(signal)):
        curr_buffer = signal[i - fltr_buffer_size:i]

        curr_analytic = echt(curr_buffer, b, a, fs_filter)
        curr_phase = np.angle(curr_analytic)[-1]
        phase_history.append(curr_phase)

        # at trough, phase jumps from 2pi to 0
        # phase should be strictly increasing everywhere else
        if (prev_phase - curr_phase) > np.pi:
            # register delayed stim event
            delayed_idx = i + random.sample(range(max_delay), 1)[0]
            if delayed_idx < len(signal):
                event_index.append(delayed_idx)

        prev_phase = curr_phase

    return np.array(event_index), np.array(phase_history)


def hilbert_detection(signal, target_phase, edge_fix=None, filter_lowcut=6, filter_highcut=9,
                      fs_filter=1500, fltr_buffer_size=400, fltr_order=2):
    """

    Parameters
    ----------
    signal
    target_phase
    edge_fix
    filter_lowcut
    filter_highcut
    fs_filter
    fltr_buffer_size
    fltr_order

    Returns
    -------

    """
    event_index = []
    phase_history = [-1]*fltr_buffer_size
    stim_ok = True
    prev_phase = -1

    butter_filter = butter(fltr_order, np.array([filter_lowcut, filter_highcut])/(fs_filter/2),
                           btype='bandpass', output='sos')

    # print(signal[:20])
    # raise Exception('breakpoint')

    for i in range(fltr_buffer_size, len(signal)):
        curr_buffer = signal[i-fltr_buffer_size:i]

        ### options to deal with edge distortions
        if edge_fix == 'mirror':
            # mirror method: flip the buffer and 3connect tail-to-tail
            curr_buffer = np.append(curr_buffer, np.flip(curr_buffer))
        elif edge_fix == 'autoregressive':
            # autoregressive model: use pre-trained AR model to forecast
            pass
        
        filtered = sosfiltfilt(butter_filter, curr_buffer)
        curr_phase = np.angle(hilbert(filtered))[-1]
        phase_history.append(curr_phase)

        # at trough, phase jumps from 2pi to 0
        # phase should be strictly increasing everywhere else
        if (prev_phase - curr_phase) > np.pi:
            stim_ok = True

        prev_phase = curr_phase

        if curr_phase >= target_phase and stim_ok:
            event_index.append(i)
            stim_ok = False

    return np.array(event_index), np.array(phase_history)


def force_reset(sample_count, reset_on, reset_threshold):
    try:
        return reset_on and sample_count >= reset_threshold
    except TypeError:
        return False


def phase_mapping_model_free(derivative_history, target_phase,
                             fltr_buffer_size=400,
                             default_slope=0.012,
                             num_to_wait=3,
                             derv_bar=2,
                             gradient_factor=1,
                             reset_on=False, reset_threshold=250,
                             lock_on=False, lockdown=50):
    """

    Parameters
    ----------
    derivative_history : array, a simulated time series of online estimated derivative
    time : array, raw time array imported from trodes. its length should be exactly fltr_buffer_size shorter
    than derivative_history
    fltr_buffer_size : int, size of data array taken for filtering
    target_phase : float, target phase to detect
    default_slope : float, slope of phase vs. sample count relation. used for linear interpolation
    num_to_wait : int, number of samples to exceed derivative magnitude threshold before flipping sign
    derv_bar : derivative magnitude threshold to exceed at critical points to trigger sign flip
    gradient_factor : float, multiplication factor to increase phase vs. sample_count slope for rising half cycle
    reset_on: boolean, whether to reset curr_phase (phase estimation)
    reset_threshold: int, number of samples to trigger reset
    lock_on: boolean, whether to trigger lockdown period after each trough detection
    lockdown: int, length of lockdown period

    Returns
    -------
    event_index : array of index of the simulated detection events in the raw lfp/time/timestamp array
    event_time : array of time in seconds of the simulated detection events
    phase_history : array, time series of online estimated phase
    """
    event_index = []
    # to make phase_history the same length as everything else
    # prepend with impossible phase values
    phase_history = [-1]*fltr_buffer_size

    curr_sign = derivative_history[0] > 0
    sign_buffer = [curr_sign] * num_to_wait
    curr_phase = None
    sample_count = None
    stim_ok = True
    slope = default_slope

    in_lock = 0

    for i, curr_derv in enumerate(derivative_history):

        if in_lock != 0:
            in_lock -= 1

        ### ------- PHASE INTERPOLATION -------
        try:
            curr_phase = sample_count * slope
            phase_history.append(curr_phase)
        except TypeError:
            # sample count is None initially
            # pass
            phase_history.append(-1)

        ### ------- UPDATE SIGN BUFFER -------
        sign_buffer.append(curr_derv > 0)
        sign_buffer.pop(0)

        ### ------- INCREMENT SAMPLE CNT -------
        try:
            sample_count += 1
        except TypeError:
            pass

        if_flip = curr_sign + sum(sign_buffer) / num_to_wait == 1 and np.abs(curr_derv) >= derv_bar
        if_force = force_reset(sample_count, reset_on, reset_threshold)

        ### ------- CRITICAL POINT / RESET -------
        if (if_flip or if_force) and in_lock == 0:

            # correct phase
            curr_phase = curr_sign * np.pi

            # update slope
            if sample_count is not None:
                if curr_sign:
                    # at peaks
                    slope = (2 - curr_sign) * np.pi / sample_count
                else:
                    # at troughs
                    slope = (2 - curr_sign) * np.pi / (sample_count / gradient_factor)
                    # enter lockdown period
                    in_lock = lockdown * lock_on

                # map 1 (positive) to pi, 0 (negative) to 2pi
                sample_count = curr_sign * int(np.pi / slope)
                # only reset sample count to 0 at phase = 2pi

            else:
                # initialize sample count according to current phase (0 or pi)
                sample_count = curr_sign * int(np.pi / slope)

            # reset stim_ok
            if not curr_sign:
                stim_ok = True

            # flip sign
            curr_sign = not curr_sign
            # necessary for force reset
            sign_buffer = [curr_sign] * num_to_wait

        ### ------- DETECT TARGET PHASE -------
        try:
            if curr_phase >= target_phase and stim_ok:
                # (+ fltr_buffer_size) to align index with other data arrays
                event_index.append(i + fltr_buffer_size)
                stim_ok = False
        except TypeError:
            # curr_phase is None initially
            pass

    return np.array(event_index), np.array(phase_history)


def get_vonmises_random(phase_center, phase_kappa, sample_count=250):
    rand_phase = np.random.vonmises(phase_center, phase_kappa)
    rand_phase += 2*np.pi*(rand_phase < 0)
    return translate_phase_to_sample_count(rand_phase, sample_count)


def translate_phase_to_sample_count(rand_phase, sample_count):
    return int(rand_phase*sample_count/(2*np.pi))