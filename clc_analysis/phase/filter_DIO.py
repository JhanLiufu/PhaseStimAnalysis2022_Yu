import numpy as np
import pandas as pd
from clc_analysis.phase.replace_with_nearest import replace_with_nearest


def filter_dio(list_dio_name, raw_dio_dict, time_dict, fs_time):
    """
    Organize DIO data by tetrode and session

    Parameters
    ----------
    list_dio_name : list of strings of DIO name
    raw_dio_dict : raw DIO event dictionary that lists DIO event in chronological order. refer to
    Analysis_clc_Phase_event_Jhan to see what they are
    time_dict : dictionary of time data (time and timestamp)
    fs_time : int, the number of trodes timestamp that one second is equal to

    Returns
    -------
    dictionary of DIO data organized by tetrode and session
    """
    filtered_dio_dict = {}
    for dio_name in list_dio_name:

        individual_dio_dict = {}

        for ses_name, df_curr in raw_dio_dict.items():

            try:
                df_curr_fltr = df_curr[df_curr.dio_bit == dio_name]

                on_timestamp = np.array(df_curr_fltr.get('on_time'))
                off_timestamp = np.array(df_curr_fltr.get('off_time'))
                on_time = (on_timestamp - time_dict.get(ses_name).get('timestamp')[0]) / fs_time
                off_time = (off_timestamp - time_dict.get(ses_name).get('timestamp')[0]) / fs_time

                # length = np.array(df_curr_fltr.get('duration')/np.timedelta64(1, 's'))*1000

                individual_dio_dict.update({ses_name: pd.DataFrame({'on_timestamp': on_timestamp,
                                                                    'on_time': on_time,
                                                                    'off_timestamp': off_timestamp,
                                                                    'off_time': off_time,
                                                                    # 'length': length
                                                                    })})

            except AttributeError:
                pass

        filtered_dio_dict.update({dio_name: individual_dio_dict})

    return filtered_dio_dict


def filter_dio_session_data(filtered_dio_dict, session_name):
    """
    Select DIO data in the specified session

    Parameters
    ----------
    filtered_dio_dict : refer to filter_DIO.filter_dio
    session_name : string, date+session number, 20230725_01 for example

    Returns
    -------
    dictionary of session DIO data organized by tetrode
    """
    dio_session_dict = {}
    for dio_curr, individual_dio_dict in filtered_dio_dict.items():
        dio_session_dict.update({dio_curr: individual_dio_dict.get(session_name)})

    return dio_session_dict


def filter_pump_event(SessionData, pump_name, duration_filter=False, duration_bar=None, sensor_filter=False,
                      sensor_name=None, replace=False):
    """
    Select pump events that:
    (1) lasted longer than certain value
    (2) didn't happen during certain sensor events

    Parameters
    ----------
    SessionData : SessionData class object which contains data of a session
    pump_name : string of DIO pump name
    duration_filter : boolean, whether to filter events by duration
    duration_bar : int, events that lasted longer than this will be included
    sensor_filter : boolean, whether to filter out events during sensor events
    sensor_name : list of strings of sensor name, pump events during whose events will be excluded
    replace : whether to replace the filtered pump event timestamps with the nearest LFP timestamps

    Returns
    -------
    pump_on_time : array of time points in seconds of the selected pump events
    pump_on_timestamp : array of timestamps that correspond to pump_on_time
    pump_length: array of pump event duration of the included event
    """
    df_pump = SessionData.pump_data.get(pump_name)

    if duration_filter:
        try:
            df_pump = df_pump[df_pump['length'] >= duration_bar]
        except TypeError:
            print('duration filter failed; specify duration threshold')

    pump_on_time = np.array(df_pump.get('on_time'), dtype=float)
    pump_on_timestamp = np.array(df_pump.get('on_timestamp'), dtype=int)
    pump_on_length = np.array(df_pump.get('length'), dtype=float)

    if sensor_filter:

        reward_period = get_sensor_on_period(SessionData, sensor_name)
        pump_incl = np.invert(np.isin(pump_on_timestamp, reward_period))
        pump_on_time = pump_on_time[pump_incl]
        pump_on_timestamp = pump_on_timestamp[pump_incl]
        pump_on_length = pump_on_length[pump_incl]

    if replace:
        pump_on_timestamp = replace_with_nearest(pump_on_timestamp, SessionData.timestamp)

    return pump_on_time, pump_on_timestamp, pump_on_length


def get_sensor_on_period(SessionData, sensor_name):
    """
    Generate an array of timestamps when the specified sensors are on

    Parameters
    ----------
    SessionData : SessionData class object which contains data of a session
    sensor_name : list of strings of sensor name

    Returns
    -------
    sensor_on_period: array-like
    """
    sensor_on_timestamp = np.array([])
    sensor_off_timestamp = np.array([])
    try:
        for s in sensor_name:
            sensor_on_timestamp = np.append(sensor_on_timestamp,
                                            SessionData.sensor_data.get(s).get('on_timestamp'))
            sensor_off_timestamp = np.append(sensor_off_timestamp,
                                             SessionData.sensor_data.get(s).get('off_timestamp'))

        # this only works when sensor events from multiple sensors do not overlap
        sensor_on_timestamp = np.sort(sensor_on_timestamp)
        sensor_off_timestamp = np.sort(sensor_off_timestamp)

        sensor_on_period = np.array([])
        for start, end in zip(sensor_on_timestamp, sensor_off_timestamp):
            sensor_on_period = np.hstack([sensor_on_period, np.linspace(start, end, (end - start + 1))])

    except TypeError:
        print('sensor period exclusion failed; specify sensor name')

    return sensor_on_period



