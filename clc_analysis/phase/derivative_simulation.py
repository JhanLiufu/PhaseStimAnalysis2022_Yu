import numpy as np
from scipy.signal import sosfiltfilt, butter


def generate_matrix(size):
    """
    Generate matrix for linear regression (linear algebra method)

    Parameters
    ----------
    size : int, size of matrix created for regression

    Returns
    -------
    A : 2-dimensional numpy array that represents matrix for linear regression
    """
    sampling_axis = np.arange(size)
    A = np.vstack([sampling_axis, np.ones(len(sampling_axis))]).T
    return A


def calculate_derv(A, buffer):
    """
    Calculate derivative / linear regression with linear algebra

    Parameters
    ----------
    A : matrix for linear regression. see generate_matrix()
    buffer : list or array of values for linear regression

    Returns
    -------
    float, derivative / slope of the regression line
    """
    curr_regr = buffer[:, np.newaxis]
    pinv = np.linalg.pinv(A)
    alpha = pinv.dot(curr_regr)
    return alpha[0][0]


def derivative_history(raw_data, fltr_buffer_size=400, regr_buffer_size=50,
                       filter_lowcut=7, filter_highcut=9, fs_filter=1500,
                       edge_fix=None, order=None):
    """
    Simulate online, stepwise calculation of derivative

    Parameters
    ----------
    SessionData : SessionData class object which contains data of a session
    fltr_buffer_size : int, size of data array taken for filtering
    regr_buffer_size : int, size of filtered data array taken for regression
    filter_lowcut : float, lower bound of the frequency range that LFP data will be filtered into
    filter_highcut : float, upper bound of the frequency range that LFP data will be filtered into
    fs_filter : int, sampling frequency of LFP signal
    edge_fix : string, method to deal with edge distortion of online filtering
    order : int, # of order of autoregressive model if that method is chosen

    Returns
    -------
    a simulated time series of online estimated derivative
    """
    if edge_fix == 'ar' and order == None:
        raise ValueError("Input order for autoregressive model")

    butter_filter = butter(1, [filter_lowcut, filter_highcut], 'bp', fs=fs_filter, output='sos')
    derivative_history = []
    A = generate_matrix(regr_buffer_size)

    # temporary print statement
    # print(raw_data[:20])
    # raise Exception('breakpoint')

    for i in range(fltr_buffer_size, len(raw_data)):
        curr_buffer = raw_data[i - fltr_buffer_size:i]
        if edge_fix == 'mirror':
            curr_buffer = np.append(curr_buffer, np.flip(curr_buffer))
            curr_filtered = sosfiltfilt(butter_filter, curr_buffer)
            half_len = int(len(curr_buffer)/2)
            curr_filtered = curr_filtered[:half_len]
        elif edge_fix == 'ar':
            pass
        else:
            curr_filtered = sosfiltfilt(butter_filter, curr_buffer)

        curr_derv = calculate_derv(A, curr_filtered[-regr_buffer_size:])
        derivative_history.append(curr_derv)

    return np.array(derivative_history)