# PhaseStimAnalysis2022_Yu
Package for analyzing performance of phase detection and phase-specific stimulation systems. For more introduction of the project, see its [project page](https://jhanliufu.github.io/projects/closed_loop_control.html) on Jhan's website.

## Installation
Clone or download this repository, ```cd``` into the repository and run ```pip install .```

## Files
- **[session_preprocessing.py](clc_analysis/phase/session_preprocessing.py)** defines the ```SessionParam``` and ```SessionData``` objects. ```SessionParam``` contains the meta-parameters of an experiment session. ```SessionData``` preprocesses and integrates all data belong to the session specified by ```SessionParam```. The data of interest is local field potential (LFP) data and digital input and output (DIO) data.