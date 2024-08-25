"""
session_preprocessing: methods to collect all raw data from a session and perform preliminary analysis
"""
import numpy as np
import pandas as pd
from scipy.signal import hilbert, butter, sosfiltfilt
from clc_analysis.phase.filter_DIO import filter_dio_session_data


def read_target(session_param, target_type):
    """
    Helper function of SessionParam; helps recognize and convert the detection or stimulation target read from input
    parameter spreadsheet

    Parameters
    ----------
    session_param : pandas dataframe whose rows contain parameter information of a session
    target_type : string, 'Stimulation_target' or 'Detection_target'

    Returns
    -------
    float target phase value
    """
    target_raw = session_param.get(target_type).values[0]

    if type(target_raw) is not str:
        return target_raw

    pi_idx = target_raw.index('np.pi')
    try:
        try:
            numerator = int(target_raw[0:pi_idx-1])
        except IndexError:
            numerator = 1

        try:
            denominator = int(target_raw[pi_idx+6:])
        except IndexError:
            denominator = 1

        target_numerical = numerator*np.pi/denominator
    except ValueError:
        target_numerical = None
        print('Incorrect target format; write target in terms of PI or 0')

    return target_numerical


class SessionParam:
    def __init__(self, session, day_param=None, date=None, dir_config=None, config=None, task=None, tetrode=None):
        """
        SessionParam class constructor, organizes session parameters into an object

        Parameters
        ----------
        day_param : pandas dataframe whose rows contain session parameter information
        session : int session id, used to select parameters of the session of interest from day_param
        dir_config: string, location of config file
        config:
        task:
        tetrode:
        """
        self.session_num = session

        if day_param is not None:
            session_param = day_param[day_param['Session'] == session]

            # SessionParam should at least have date, session, track, and config
            self.date = str(session_param['Date'].values[0])
            self.config = session_param['Config_file'].values[0]
            self.task = session_param['Task'].values[0]
            self.tetrode = int(session_param.get('Tetrode'))

        else:
            self.date = date
            self.config = config
            self.task = task
            self.tetrode = tetrode

        self.session_name = f'{self.date}_{self.session_num:02d}'

        # get laser pump id and sensor ids from config file
        config = pd.read_excel(dir_config + self.config, engine='openpyxl')
        task_config = config[config['Track'] == self.task]

        try:
            self.stim_pump_id = task_config['out laser pump'].values[0]
            self.stim_pump_name = f'ECU_Dout{self.stim_pump_id}'
        except KeyError:
            pass

        dio_tags = task_config.columns.to_numpy()
        sensor_tags = dio_tags[['in reward' in d for d in dio_tags]].tolist()
        self.sensor_id = [task_config[s].values[0] for s in sensor_tags]
        self.sensor_name = [f'ECU_Din{s_id}' for s_id in self.sensor_id]


class SessionData:
    def __init__(self, SessionParam, lfp_data=None, time_dict=None, pump_dict=None, sensor_dict=None,
                 filter_lowcut=None, filter_highcut=None, fs_filter=None, estimated_phase=None, tetrode=None,
                 amplitude_factor=1):
        """
        SessionData class constructor, organizes session data into an object

        Parameters
        ----------
        SessionParam : SessionParam class object, which contains session parameter info
        lfp_data : dictionary of raw LFP data imported from trodes
        time_dict : dictionary of time data (time and timestamp)
        pump_dict : dictionary of pump data (refer to filter_DIO.filter_dio)
        sensor_dict : dictionary of sensor data (refer to filter_DIO.filter_dio)
        filter_lowcut : float, lower bound of the frequency range that LFP data will be filtered into
        filter_highcut : float, upper bound of the frequency range that LFP data will be filtered into
        fs_filter : int, sampling frequency of LFP signal
        estimated_phase: array like, estimated phases by algorithm simulation
        amplitude_factor: amplitude multiplication factor, for simulation use
        """
        self.amplitude_factor = amplitude_factor

        if lfp_data is not None:
            try:
                if tetrode is None:
                    tetrode = SessionParam.tetrode
                butter_filter = butter(1, [filter_lowcut, filter_highcut], 'bp', fs=fs_filter, output='sos')
                self.raw = lfp_data[tetrode][SessionParam.session_num]*self.amplitude_factor
                self.filtered = sosfiltfilt(butter_filter, self.raw)
                self.gradient = np.gradient(self.filtered)
                self.analytical = hilbert(self.filtered)
                self.envelope = np.abs(self.analytical)
                self.phase = np.angle(self.analytical)
            except TypeError:
                print('specify filter_lowuct, filter_highcut, and fs_filter')
        else:
            self.raw = None
            self.filtered = None
            self.gradient = None
            self.analytical = None
            self.envelope = None
            self.phase = None

        if time_dict is not None:
            self.time = time_dict.get(SessionParam.session_name).get('time')
            self.timestamp = time_dict.get(SessionParam.session_name).get('timestamp')
        else:
            self.time = None
            self.timestamp = None

        if pump_dict is not None:
            self.pump_data = filter_dio_session_data(pump_dict, SessionParam.session_name)
        else:
            self.pump_data = None

        if sensor_dict is not None:
            self.sensor_data = filter_dio_session_data(sensor_dict, SessionParam.session_name)
        else:
            self.sensor_data = None

        if estimated_phase is not None:
            self.estimated_phase = estimated_phase
        else:
            self.estimated_phase = None


