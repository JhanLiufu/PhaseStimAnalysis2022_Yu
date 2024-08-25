### ----------------------------- IMPORT PACKAGES -----------------------------
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import itertools
import pickle as pkl
from scipy.signal import sosfiltfilt, butter, hilbert
from scipy import stats

from emk_analysis import builder_experiment as bld_exp
from emk_neuro_analysis.lfp import iterator as lfp_iter
from emk_neuro_analysis.dio import func as dio_func
from emk_analysis import iterator as emk_iter
from pathlib import Path

from clc_analysis.phase import organize_cycle as oc
from clc_analysis.phase import filter_DIO as fd
from clc_analysis.phase.filter_DIO import filter_pump_event
from clc_analysis.phase.phase_IO import read_day_param
from clc_analysis.phase.session_preprocessing import SessionData, SessionParam
from clc_analysis.phase import phase_plot as p_plot
from clc_analysis.phase.derivative_simulation import derivative_history
from clc_analysis.phase import event_detection as ed

import traceback

### READ IN SIMULATION TASK QUEUE

# data drive
data_disk = '/project/jaiyu/'

# analysis folder
analysis_folder = 'clc_manuscript/'

# simulation input file location
dir_input = f'{data_disk}Analysis/{analysis_folder}clc_simulation/Input/'
input_fname = 'clc_simulation_queue_20231102_6202868_phase_mapping_amplitude_1.xlsx'

# simulation results location
dir_results = f'{data_disk}Analysis/{analysis_folder}clc_simulation/Results/'
results_folder_name = '20231102_6202868_phase_mapping_amplitude_1/'
dir_session_results = f'{dir_results}{results_folder_name}'
os.makedirs(dir_session_results, exist_ok = True)

# read in input file
task_queue = pd.read_excel(f'{dir_input}{input_fname}', engine='openpyxl')
# print('task queue read')

for _, row in task_queue.iterrows():

    try:
        ### ---------------------- SPECIFY SIMULATION PARAMETERS - READ FROM INPUT SPREADSHEET ----------------------
        
        # name of experiment
        experiment_name = str(int(row.experiment_name))

        experiment_phase = row.experiment_phase
        
        session = int(row.session)
        
        date = str(int(row.date))
        
        tetrode = int(row.tetrode)

        ### ----------------------------- CREATE RESULTS FOLDER FOR INDIVIDUAL TASK -----------------------------
        if row.detect:
            if row.algorithm == 'phase_mapping':
                task_results_folder = (f'{dir_session_results}{experiment_name}_'
                                       f'{date}_s{session}_tet{tetrode}_{row.algorithm}_n{int(row.num_to_wait)}'
                                       f'_r{int(row.reset_on)}_rt{int(row.reset_threshold)}_l{int(row.lock_on)}'
                                       f'_ld{int(row.lockdown)}_d{int(row.derv_bar)}_g{row.gradient_factor}'
                                       f'_a{row.amplitude_factor}/')
            else:
                task_results_folder = (f'{dir_session_results}{experiment_name}_'
                                       f'{date}_s{session}_{row.algorithm}_a{row.amplitude_factor}/')
        elif row.signal:
            task_results_folder = (f'{dir_session_results}{experiment_name}_'
                                   f'{date}_s{session}_tet{tetrode}_signal_property/')
        else:
            continue
        os.makedirs(task_results_folder, exist_ok=True)
        # print(f'folder created: {task_results_folder}')
        
        # directory with the preprocessed/extracted data files
        dir_preprocess = f'{data_disk}/Data/{experiment_name}/preprocessing/'

        # print(1);
        
        # Location of track config file. 
        # This is an excel spreadsheet that specifies the identities of the DIO for your experiment.
        dir_config = f'{data_disk}/Data/{experiment_name}/config/'
        fname_config_track = (f'{data_disk}/Data/{experiment_name}/config/{row.config_file}')

        # print(2);
        
        # Location of session parameter file
        # This is an excel spreadsheet that specifies the CLC parameters used in your experiment
        # fname_session_parameter = (f'{data_disk}/Data/{experiment_name}/config/6082755_session_parameter_record.xlsx')
        fname_session_parameter = (f'{data_disk}/Data/{experiment_name}/metadata/clc_session_parameter_record.xlsx')

        # print(3);
        
        # Location of day records. 
        # This is an excel spreadsheet that lists details for each session on your experiment day.
        dir_records = (f'{data_disk}/Data/{experiment_name}/dayrecords/')

        # print(5);
        
        # chose the date - as a list
        choose_dates = [date,]

        # print(6);
        
        # choose the epoch - as a list
        epoch_list = [session, ]

        # print(7);
        
        # choose the tetrodes - as a list
        tet_list = [tetrode,]

        # print(8);
        
        ### ----------------------------- BUILD DAY RECORDS -----------------------------
        dayrecord_postfix = row.dayrecord_postfix
        if row.dayrecord_postfix == '/':
            dayrecord_postfix = ''

        dict_sessions_all = bld_exp.build_experiment_from_file(experiment_name, choose_dates, 
                                                               dir_records, fname_config_track,
          dayrecord_file_postfix=f'_{experiment_name}_training_record{dayrecord_postfix}.xlsx')

        # print(9);
        
        # print(dir_preprocess)
        # print(tet_list)
        # print(choose_dates)
        # print(epoch_list)
        
        ### ----------------------------- IMPORT LFP DATA ----------------------------- 
        lfp_data, lfp_timestamp, _ = lfp_iter.iterate_lfp_load(dir_preprocess,
                                                               tet_list,
                                                               choose_dates,
                                                               epoch_list=epoch_list, 
                                                               remove_movement_artifact=False,
                                                               filter_linenoise=True,
                                                               print_debug=False)
        # lfp_func.concatenate_lfptime(lfp_timestamp, df_epoch_lfp)

        # print(10);
        
        ### ----------------------------- TRANSFORM TIME -----------------------------
        time_dict = {}
        fs_time = 30000
        fs_filter = 1500
        for ses_num, time_curr in lfp_timestamp.items():
            full_session_name = f'{choose_dates[0]}_0{ses_num}'
            time_dict.update({full_session_name:{'timestamp':np.array(time_curr),
                                                 'time':(np.array(time_curr) - time_curr[0])/fs_time}})

        ### ----------------------------- IMPORT DIO -----------------------------
        _, df_sensor, _ = dio_func.get_DIO_events(experiment_name, experiment_name, 
                                                  dict_sessions_all, sensor_pump_map=[],
                                                  select_task=[],  filter_retrigger=0, 
                                                  trodes_version=2)
        dict_dio_in = dio_func.sort_trials_to_sessions(df_sensor, dict_sessions_all, add_date_epoch=True)

        ### ----------------------------- ORGANIZE SESSIONDATA -----------------------------
        ### access parameters in a day
        # day_param = read_day_param(fname_session_parameter, choose_dates)
        ### select session for analysis
        session_param = SessionParam(session,
                                     date=str(date),
                                     dir_config=dir_config,
                                     config=row.config_file,
                                     task=row.task,
                                     tetrode=tetrode)

        reward_dict = fd.filter_dio(session_param.sensor_name, dict_dio_in, time_dict, fs_time)

        ### offline simulation
        if row.detect:
            # amplitude_factor = row.amplitude_factor
            sig_input = lfp_data.get(session_param.tetrode).get(session_param.session_num)*row.amplitude_factor
            if row.algorithm == 'phase_mapping':
                derv = derivative_history(sig_input)
                simulated_idx, simulated_phase = ed.phase_mapping_model_free(derv, row.detection_target, 
                                                                             reset_on=row.reset_on, 
                                                                             reset_threshold=row.reset_threshold,
                                                                             lock_on=row.lock_on, 
                                                                             lockdown=row.lockdown, 
                                                                             derv_bar=row.derv_bar,
                                                                             gradient_factor=row.gradient_factor)
            elif row.algorithm == 'HT':
                simulated_idx, simulated_phase = ed.hilbert_detection(sig_input, row.detection_target)

            elif row.algorithm == 'ecHT':
                simulated_idx, simulated_phase = ed.endpoint_correcting_hilbert_transform(sig_input, row.detection_target)
        else:
            simulated_idx = None
            simulated_phase = None

        ### collect and preprocess data of session
        session_data = SessionData(session_param, 
                                   lfp_data=lfp_data, 
                                   time_dict=time_dict, 
                                   sensor_dict=reward_dict, 
                                   filter_lowcut=7, 
                                   filter_highcut=9, 
                                   fs_filter=fs_filter,
                                   estimated_phase=simulated_phase
                                  )

        ### ----------------------------- ORGANIZE CYCLES -----------------------------
        if row.detect:
            df_cycles = oc.organize_cycles(session_data,
                                           event_time=session_data.time[simulated_idx], 
                                           event_timestamp=session_data.timestamp[simulated_idx],
                                           label_in_reward=True,
                                           reward_sensor_name=session_param.sensor_name,
                                           label_after_first_event=True
                                           )
            # Idenfity cycles valid cycles
            df_valid_cycles = df_cycles[df_cycles['in_reward']==False]
            df_valid_cycles = df_valid_cycles[df_valid_cycles['after_first_event']==True]
        else:
            df_cycles = oc.organize_cycles(session_data,
                                           label_in_reward=True,
                                           reward_sensor_name=session_param.sensor_name,
                                           label_after_first_event=False
                                           )
            # Idenfity cycles valid cycles
            df_valid_cycles = df_cycles[df_cycles['in_reward']==False]

        df_valid_cycles = df_valid_cycles.reset_index()
        # save df_cycles
        df_cycles.to_csv(f'{task_results_folder}df_cycles.csv', encoding='utf-8', index=False)

        ### ----------------------------- ANALYSIS -----------------------------
        if row.signal:
            valid_sig = np.hstack(df_valid_cycles.filtered)
            dict_power_dist = p_plot.power_distribution([np.abs(hilbert(valid_sig))], 
                                                        ['valid'],
                                                        session_param=session_param,
                                                        dir_fig=task_results_folder)
            dict_cyc_asym = p_plot.rising_falling_lin_reg(valid_sig, 
                                                          session_param=session_param,
                                                          dir_fig=task_results_folder)

            with open(f'{task_results_folder}dict_power_dist.pkl', 'wb') as f_dpd:
                pkl.dump(dict_power_dist, f_dpd)

            with open(f'{task_results_folder}dict_cyc_asym.pkl', 'wb') as f_dca:
                pkl.dump(dict_cyc_asym, f_dca)

        if not row.detect:
            # skip analyses (stim events) below
            continue

        # dict_metrics holds numerical metrics
        dict_metrics = {}

        # average estimation error by phase
        dict_estm_error = p_plot.estm_phase_error_dist([np.hstack(df_valid_cycles.estimated_phase)],
                                                       ['valid'],
                                                       np.hstack(df_valid_cycles.offline_phase),
                                                       session_param=session_param,
                                                       dir_fig=task_results_folder)

        avg_err = np.mean(np.abs(np.hstack(df_valid_cycles.offline_phase) - np.hstack(df_valid_cycles.estimated_phase)))
        dict_metrics.update({'avg_estm_err': avg_err})

        # phase distribution
        dict_event_phases = p_plot.event_phase_hist([np.hstack(df_valid_cycles.event_phase)], 
                                                    ['valid'], 
                                                    width=0.2,
                                                    session_param=session_param,
                                                    dir_fig=task_results_folder)
        event_phases = dict_event_phases.get('valid').get('phases')
        dict_metrics.update({'phase_avg': stats.circmean(event_phases),
                             'phase_var': stats.circvar(event_phases)})

        # local rhythmicity / neighbor phase difference distribution
        dict_nb_p_diff = p_plot.neighbor_phase_diff_hist([p_plot.df_cycles_to_lin_phase(df_valid_cycles)], 
                                                         ['valid'], 
                                                         upper_bound=10 * np.pi, 
                                                         width=0.2,
                                                         session_param=session_param,
                                                         dir_fig=task_results_folder)
        nb_p_diff = dict_nb_p_diff.get('valid').get('neighbor_diff')
        dict_metrics.update({'nb_p_diff_var': stats.circvar(nb_p_diff, high=4*np.pi)})
        dict_metrics.update({'npvi': p_plot.calculate_npvi(nb_p_diff)})

        # global rhythmicity / stim phase autocorrelogram
        dict_phase_xcorr = p_plot.event_phase_self_xcorr([p_plot.df_cycles_to_lin_phase(df_valid_cycles)], 
                                                         ['valid'], 
                                                         upper_bound=50*np.pi, 
                                                         session_param=session_param,
                                                         dir_fig=task_results_folder)

        h = dict_phase_xcorr.get('valid').get('hist')
        h = h/sum(h)
        avg_coherence = p_plot.average_coherence(dict_phase_xcorr.get('valid').get('edges')[:-1], h)
        dict_metrics.update({'avg_coherence': avg_coherence})

        ### ----------------------------- SAVE ANALYSIS RESULTS -----------------------------
        print('Saving analysis results')
        with open(f'{task_results_folder}dict_metrics.pkl', 'wb') as f_dm:
            pkl.dump(dict_metrics, f_dm)

        with open(f'{task_results_folder}dict_event_phases.pkl', 'wb') as f_dep:
            pkl.dump(dict_event_phases, f_dep)

        with open(f'{task_results_folder}dict_nb_p_diff.pkl', 'wb') as f_npd:
            pkl.dump(dict_nb_p_diff, f_npd)

        with open(f'{task_results_folder}dict_phase_xcorr.pkl', 'wb') as f_px:
            pkl.dump(dict_phase_xcorr, f_px)
            
    except Exception as exc:
        # print(f'Task skipped due to the following exception:')
        # traceback.print_exc()
        # os.remove(task_results_folder)
        # print('Simulation result folder removed')
        continue