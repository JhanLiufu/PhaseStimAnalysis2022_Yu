import pandas as pd
import numbers
import os
import numpy as np
from numpy.random import vonmises, normal, binomial


def verify_ioi(ioi, ioi_var):
    '''
    Verify if current IOI belongs to dist with ioi_var
    Parameters

    ----------
    ioi
    ioi_var

    Returns
    -------

    '''
    return (ioi > (2 * np.pi - ioi_var)) and (ioi < (2 * np.pi + ioi_var))


def verify_npvi(pvi, npvi):
    '''
    Verify if current IOI belongs to dist with ioi_var
    Parameters
    ----------
    pvi
    npvi

    Returns
    -------

    '''
    # Unimplemented
    return True


def verify_phase(phase, phase_mean, phase_var):
    '''
    Verify if given phase sample is drawn from given vonmises

    Parameters
    ----------
    phase
    phase_mean
    phase_var

    Returns
    -------

    '''
    # some sort of one sample test, unimplemented
    return True


def generate_event_pattern(event_cnt,
                           phase_mean=np.pi, phase_var=np.pi / 2,
                           ioi_var=np.pi / 2, npvi=3,
                           skip_probs=0.02,
                           double_probs=0.001):
    '''

    Parameters
    ----------
    event_cnt: number of event generated
    phase_mean: center of vonmises distribution to draw event phases from
    phase_var: kappa of vonmises distribution to draw event phases from
    ioi_var: how variable the IOI values are allowed to be
    npvi: measure of how close nearby IOIs are
    skip_probs: probability of skipping current cycle
    double_probs:

    Returns
    -------

    '''
    cur_cnt = 0
    lin_phase_lst = []
    circ_phase_lst = []
    skip_mask = binomial(event_cnt, skip_probs)
    # double_mask = binomial(event_cnt, double_probs)
    prev_ioi = 2 * np.pi
    prev_lin_phase = phase_mean
    success = True

    while cur_cnt < event_cnt:

        # may skip current cycle
        if skip_mask[cur_cnt]:
            # if skipping, advance to next cycle directly
            cur_cnt += 1
            continue

        # generate IOI
        cur_ioi = normal(2 * np.pi, ioi_var)
        # calculate current phase
        cur_lin_phase = prev_lin_phase + cur_ioi
        cur_circ_phase = cur_lin_phase % (2 * np.pi)
        # calculate current PVI
        cur_pvi = abs((cur_ioi - prev_ioi) / ((cur_ioi + prev_ioi) / 2))
        # verify
        success = verify_phase(cur_circ_phase, phase_mean, phase_var) and verify_npvi(cur_pvi, npvi)

        # if success, advance to next cycle; if not try again
        if success:
            lin_phase_lst.append(cur_lin_phase)
            circ_phase_lst.append(cur_circ_phase)
            prev_lin_phase = cur_lin_phase
            prev_ioi = cur_ioi
            cur_cnt += 1

    return lin_phase_lst, circ_phase_lst

def get_task_queue_from_dir(dir_r):
        for f in os.listdir(dir_r):
            if f.endswith('xlsx'):
                return pd.read_excel(f'{dir_r}{f}', engine='openpyxl')


class TaskParam:
    def __init__(self, row, task_type='rodent_lfp'):
    
        if task_type == 'rodent_lfp':
            # name of experiment, could be animal id or not
            if type(row.experiment_name) == int or type(row.experiment_name) == float:
                self.exp_name = str(int(row.experiment_name))
            else:
                self.exp_name = row.experiment_name

            self.animal_id = str(int(row.animal_id))
            self.exp_phase = row.experiment_phase
            self.session = int(row.session)
            self.date = str(int(row.date))
            self.tetrode = int(row.tetrode)
            self.dayrecord_postfix = row.dayrecord_postfix
            data_info = f'{self.exp_name}_{self.animal_id}_{self.date}_s{self.session}_tet{self.tetrode}'
                        
        elif task_type == 'human_eeg':
            self.exp_name = str(row.experiment_name)
            self.subject_id = int(row.subject_id)
            self.channel = int(row.channel)
            data_info = f'{self.exp_name}_subj{self.subject_id}_tet{self.channel}'
        
        elif task_type == 'human_tremor':
            self.exp_name = str(row.experiment_name)
            self.subject_id = int(row.subject_id)
            self.condition = int(row.condition_id)
            self.repetition = int(row.repetition_id)
            data_info = f'{self.exp_name}_subj{self.subject_id}_cond{self.condition}_rep{self.repetition}'
        
        self.algorithm = row.algorithm
        self.detection_target = row.detection_target
        self.filter_lowcut = int(row.filter_lowcut)
        self.filter_highcut = int(row.filter_highcut)
        self.amplitude_factor = row.amplitude_factor

        if row.detect:
            if row.algorithm == 'phase_mapping':
                task_results_folder = (
                    f'{data_info}'
                    f'_{self.algorithm}'
                    f'_fl{self.filter_lowcut}'
                    f'_fh{self.filter_highcut}'
                    f'_t{self.detection_target}'
                    f'_n{int(row.num_to_wait)}'
                    # f'_r{int(row.reset_on)}'
                    # f'_rt{int(row.reset_threshold)}'
                    # f'_l{int(row.lock_on)}'
                    # f'_ld{int(row.lockdown)}'
                    f'_d{int(row.derv_bar)}'
                    f'_g{row.gradient_factor}'
                    f'_b{row.fltr_buffer_size}'
                    f'_a{self.amplitude_factor}/'
                )
            else:
                task_results_folder = (
                    f'{data_info}'
                    f'_{self.algorithm}'
                    f'_fl{self.filter_lowcut}'
                    f'_fh{self.filter_highcut}'
                    f'_t{self.detection_target}'
                    f'_b{row.fltr_buffer_size}'
                    f'_a{self.amplitude_factor}/'
                )
            
        elif row.signal:
            task_results_folder = f'{data_info}_signal_property/'
        else:
            task_results_folder = None

        self.dir = task_results_folder
