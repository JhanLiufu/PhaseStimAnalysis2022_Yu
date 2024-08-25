# PhaseStimAnalysis2022_Yu
Package for analyzing performance of phase detection and phase-specific stimulation systems. For more introduction of the project, see its [project page](https://jhanliufu.github.io/projects/closed_loop_control.html) on Jhan's website.

## Installation
Clone or download this repository, ```cd``` into the repository and run ```pip install .```

## Files
- **[session_preprocessing.py](clc_analysis/phase/session_preprocessing.py)** defines the ```SessionParam``` and ```SessionData``` objects. ```SessionParam``` contains the meta-parameters of an experiment session. ```SessionData``` preprocesses and integrates all data belong to the session specified by ```SessionParam```. The data of interest is local field potential (LFP) data and digital input and output (DIO) data.

- **[organize_cycle.py](clc_analysis/phase/organize_cycle.py)** implements the ```organize_cycle``` function. ```organize_cycle``` parses a session's data into oscillatory cycles to facilitate cycle-to-cycle analysis.

- **[phase_plot.py](clc_analysis/phase/phase_plot.py)** implements various plotting functions. For example, ```event_phase_hist``` produces a circular histogram of the stimulation phases. ```event_phase_self_xcorr``` produces an autocorrelogram of the unwrapped stimulation phases. Many more are available.

- **[notebooks](clc_analysis/notebooks)** contains jupyer notebooks for the commonly used analysis. ```Analysis_quantify_stimulation_accuracy.ipynb``` is the most often used one; it quantifies the stimulation performance.