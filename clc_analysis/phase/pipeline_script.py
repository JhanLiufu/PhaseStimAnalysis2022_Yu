import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import os

from emk_analysis import builder_experiment as bld_exp
from emk_neuro_analysis.lfp import iterator as lfp_iter
# from emk_neuro_analysis.position import iterator as pos_iter
# from mountainlab_pytools import mdaio
from emk_analysis import iterator as emk_iter

from clc_analysis.phase import organize_cycle as oc, filter_DIO as fd
from clc_analysis.phase.filter_DIO import filter_pump_event
from clc_analysis.phase.phase_IO import read_day_param, get_unanalyzed
from clc_analysis.phase.session_preprocessing import SessionData, SessionParam
from clc_analysis.phase import phase_plot as p_plot

'''
Experiment parameters
'''
# name of experiment
experiment_name = '6004874'

experiment_phase = 'stim'

# data drive
data_disk = '/project/jaiyu'

# directory with the preprocessed/extracted data files
dir_preprocess = f'{data_disk}/Data/{experiment_name}/preprocessing/'

# Figure folder, where you want to save the output figures. Usually in your experiment's analysis folder
dir_fig = f'{data_disk}/Analysis/{experiment_name}/clc_results/'

# Location of track config file.
# This is an excel spreadsheet that specifies the identities of the DIO for your experiment.
dir_config = f'{data_disk}/Data/{experiment_name}/config/'

# Location of day records.
# This is an excel spreadsheet that lists details for each session on your experiment day.
dir_records = f'{data_disk}/Data/{experiment_name}/dayrecords/'

# Location of session parameter file
# This is an excel spreadsheet that specifies the CLC parameters used in your experiment
fname_session_parameter = f'{data_disk}/Data/{experiment_name}/config/clc_session_parameter_record.xlsx'

sessions_unanalyzed = get_unanalyzed(fname_session_parameter, dir_fig)
dates = map(str, sessions_unanalyzed.Date.unique())

for curr_date in dates:
    '''
    ------------ access parameters in a day ------------
    '''
    day_param = read_day_param(fname_session_parameter, [curr_date])
    epoch_list = day_param.Session.unique().tolist()
    tet_list = day_param.Tetrode.unique()
    # each session can use different config files. need to change later in session specific analysis
    fname_config_track = dir_config + day_param.get('Config_file').values[0]

    '''
    ------------ Build day records from track config file and experiment file ------------
    '''
    data_days = []
    fname_day_record = f'{dir_records}{curr_date}_{experiment_name}_training_record_L17.xlsx'

    dict_sessions_day = bld_exp.build_day_from_file(experiment_name,
                                                    track_config_file=fname_config_track,
                                                    day_record_file=fname_day_record)
    data_days.append(dict_sessions_day)
    dict_sessions_all = bld_exp.build_all_sessions(data_days)

    '''
    ------------ Import LFP data ------------
    '''
    lfp_data, lfp_timestamp, _ = lfp_iter.iterate_lfp_load(dir_preprocess,
                                                           tet_list,
                                                           [curr_date],
                                                           epoch_list=epoch_list,
                                                           remove_movement_artifact=False,
                                                           filter_linenoise=True,
                                                           print_debug=False)

    '''
    ------------ Transform time ------------
    '''
    time_dict = {}
    fs_time = 30000
    fs_filter = 1500

    for i in lfp_timestamp.items():
        time_curr = i[1]
        time_dict.update({i[0]: {'timestamp': np.array(time_curr),
                                 'time': (np.array(time_curr) - time_curr[0]) / fs_time}})

    '''
    ------------ Import DIO data ------------
    '''
    plot_DIO = False

    filter_retrigger = 1000

    # time plotting settings
    tick_minutes = mdates.MinuteLocator(interval=5)
    tick_minutes_fmt = mdates.DateFormatter('%H:%M')
    tick_minor = mdates.SecondLocator(interval=10)

    # Specify parameters
    dict_sensor_pump_map = {6: {'pump': 'laser_pump'},
                            }
    list_dio = [6, ]
    y_label = ['laser', ]

    # plot each session
    # get data for each animal
    # initiate output
    dict_dio_out = {}

    dict_dio_in = {}

    for animal_id in [experiment_name, ]:

        # print(animal_id)
        cls_behavior = emk_iter.ProcessBehavior(dict_sessions_all,
                                                experiment_name, trodes_version=2)
        cls_behavior.filter_animals(animal_id)
        dict_rewards = cls_behavior.count_reward_delivered()

        if not dict_rewards:
            continue

        df_pump = cls_behavior.report_reward_delivered(remove_zeroth=False,
                                                       output_raw=False,
                                                       filter_retrigger=None)

        df_sensor = cls_behavior.report_triggers(remove_zeroth=False,
                                                 output_raw=False,
                                                 filter_retrigger=filter_retrigger)

        # get unique sessions
        sessions_unique = np.sort(df_pump['session'].unique())
        # print(sessions_unique)
        n_subplots = len(sessions_unique)

        if plot_DIO:
            fig = plt.figure(figsize=(10, n_subplots * 3 + 2))
            axs = fig.subplots(n_subplots, 1)
            if n_subplots == 1:
                axs = [axs, ]
                sessions_unique = [sessions_unique[0], ]

        else:
            axs = [0] * len(sessions_unique)

        # divide into sessions
        for sn, (ax, session) in enumerate(zip(axs, sessions_unique)):

            # get session times
            curr_start = dict_sessions_all.get(session).get('start')
            curr_end = dict_sessions_all.get(session).get('end')

            # get sensor and pump times
            df_sensor_curr = df_sensor[df_sensor['session'] == session]
            df_sensor_curr = df_sensor_curr[(df_sensor_curr['on_time_sys'] >= curr_start)
                                            & (df_sensor_curr['on_time_sys'] < curr_end)]
            dict_dio_in.update({int(session.split('_')[1]): df_sensor_curr})

            df_pump_curr = df_pump[df_pump['session'] == session]
            df_pump_curr = df_pump_curr[(df_pump_curr['on_time_sys'] >= curr_start)
                                        & (df_pump_curr['on_time_sys'] < curr_end)]
            dict_dio_out.update({int(session.split('_')[1]): df_pump_curr})

            if not plot_DIO:
                continue

            # plot DIO data for all sessions
            for i, d in enumerate(list_dio):
                yval = i + 1
                curr_pump_name = dict_sensor_pump_map.get(d).get('pump')
                df_plot_pump = df_pump_curr[df_pump_curr['dio'] == curr_pump_name]

                curr_sensor_name = dict_sensor_pump_map.get(d).get('sensor')
                df_plot_sensor = df_sensor_curr[df_sensor_curr['dio'] == curr_sensor_name]
                # plot well triggers

                for ind, row in df_plot_sensor.iterrows():
                    ax.scatter(row['on_time_sys'], yval + .3, s=25, c='k')

                for ind, row in df_plot_pump.iterrows():

                    try:
                        ax.plot([row['on_time_sys'],
                                 row['off_time_sys']], [yval + .15, yval + .15], c='r')

                    except:
                        pass

    '''
    ------------ Pre-process DIO data ------------
    '''
    # ### filter out DIO data of interest
    # pumps = ['ECU_Dout6']
    # # pumps = ['ECU_Dout6', 'ECU_Dout12', 'ECU_Dout13', 'ECU_Dout14']
    # pump_dict = fd.filter_dio(pumps, dict_dio_out, time_dict, fs_time)
    #
    # reward_sensors = ['ECU_Din5', 'ECU_Din4', 'ECU_Din7']
    # reward_dict = fd.filter_dio(reward_sensors, dict_dio_in, time_dict, fs_time)

    '''
    ------------ Session Data ------------
    '''
    for session in epoch_list:
        session_param = SessionParam(day_param, session, dir_config)

        session_dir = session_param.date + f'_s{session}'
        session_full_path = os.path.join(dir_fig, session_dir)
        try:
            os.mkdir(session_full_path)
        except FileExistsError:
            print(session_dir + ' analyzed')
            continue

        pump_dict = fd.filter_dio([session_param.stim_pump_name], dict_dio_out, time_dict, fs_time)
        reward_dict = fd.filter_dio(session_param.sensor_name, dict_dio_in, time_dict, fs_time)

        ### collect and preprocess data of session
        session_data = SessionData(session_param,
                                   lfp_data=lfp_data,
                                   time_dict=time_dict,
                                   pump_dict=pump_dict,
                                   sensor_dict=reward_dict,
                                   filter_lowcut=7, filter_highcut=9, fs_filter=fs_filter)

        ### stim time
        stim_time, stim_timestamp, _ = filter_pump_event(session_data, session_param.stim_pump_name,
                                                         duration_filter=False,
                                                         sensor_filter=False,
                                                         sensor_name=session_param.sensor_name)

        ### identify stim periods
        stim_pivots = np.where(np.diff(np.append(-19, stim_time)) > 1)[0][1:]
        stim_timestamp_cut = np.split(stim_timestamp, stim_pivots)
        stim_start = np.array([s[0] for s in stim_timestamp_cut])
        stim_end = np.array([s[-1] for s in stim_timestamp_cut])

        '''
        Organize data by oscillatory cycles
        '''
        df_cycles = oc.organize_cycles(session_data,
                                       event_time=stim_time,
                                       event_timestamp=stim_timestamp,
                                       label_in_reward=True,
                                       reward_sensor_name=session_param.sensor_name,
                                       label_after_first_event=True
                                       )

        # df_valid_cycles = df_cycles[df_cycles['in_reward'] == False]
        # df_valid_cycles = df_valid_cycles[df_valid_cycles['after_first_event'] == True]
        # oc.cycles_info(df_valid_cycles, 'stim')

        '''
        neighbor phase difference distribution plot
        '''
        p_plot.neighbor_phase_diff_hist([df_cycles], ['all'], session_param, upper_bound=10 * np.pi, width=0.2,
                                        dir_fig=session_full_path)

        '''
        event length distribution
        '''
        p_plot.stim_len_hist(session_data, session_param, session_param.stim_pump_name, upper_bound=100,
                             dir_fig=session_full_path)

        '''
        event time (aligned to cycle start) distribution
        '''
        p_plot.event_aligned_time_hist([df_cycles], ['all'], session_param, dir_fig=session_full_path)

        '''
        event phase distribution
        '''
        p_plot.event_phase_hist([df_cycles], ['all'], session_param, width=0.2, dir_fig=session_full_path)