# larva-tagger-tune

## Project Description
This project is part of the [LarvaTagger](https://gitlab.pasteur.fr/nyx/larvatagger.jl) project, aiming to monitor the behavior of *Drosophila* larvae in video recordings.
This repository contains the code for the automated hyper-parameter optimization of larva tracker.
Currently, three trackers are supported:
- MWT (Multi-Worm Tracker, [Paper](https://doi.org/10.1038%2Fnmeth.1625))
- Tierpsy Tracker ([Paper](https://doi.org/10.1038%2Fs41592-018-0112-1))
- WF-NTP (Wide Field-of-view Nematode Tracking Platform, [Paper](https://doi.org/10.1038/s41596-020-0321-9))

## Installation
Requires the tracker that should be used for the optimization to be installed.
We tested the code with the MWT, Tierpsy Tracker and WF-NTP.
We used the following versions of the respective trackers which all come with installation instructions:
- MWT: [GitLab Repo](https://gitlab.com/larvataggerpipelines/mwt-cli)
- Tierpsy Tracker: [GitLab Repo](https://gitlab.com/larvataggerpipelines/tierpsy-cli)
- WF-NTP: [GitHub Repo](https://github.com/Lilly-May/wf-ntp-cli)

The code was tested using a conda environment with Python 3.10.12. We tested it on MacOS and Windows machines.

The following packages are required:
matplotlib, numpy, pandas, optuna, joblib

Alternatively, a `requirements.txt` file is provided to allow reproducible installation with pip. larva-tagger-tune can be installed in a virtual environment with the following steps (for Unix-like systems):

```
git clone https://github.com/Lilly-May/larva-tagger-tune
python3.10 -m venv larva-tagger-tune
cd larva-tagger-tune
source bin/activate
pip install -r requirements.txt
```

Windows users will have to type `Scripts/activate.bat` (if in cmd) as an activation command instead of `source bin/activate`.

## Usage
The code can be run from the command line. An example call is:

```
python ParameterOptimization.py MWT /Path/to/WorkingDir /Path/to/VideoDir /Path/to/target_larvae_nr.csv --plot --nr_processes 3
```

The script requires four positional arguments:
- Tracker: The tracker that should be used for the optimization. Currently, MWT, Tierpsy Tracker and WF-NTP are supported.
- WorkingDir: The path to the working directory. This directory is expected to have a subdirectory TRACKER-cli with
the respective tracker installation.
- VideoDir: The path to the directory containing the videos that should be analyzed.
- TargetFile: The path to the .csv file containing the target number of larvae for each video
(see below for detailed description).

Additionally, the optimization process can be customized by specifying optional arguments.
You can get a list and description of all optional arguments by calling:
```
python ParameterOptimization.py --help
```


### Target Number File
The target number file is a .csv file containing the target number of larvae for each video. Since is it possible that
larvae leave the frame during the video recording, the target number of larvae can change over time. Therefore, the
target number file contains the target number of larvae for each second of the video.

The file has to be in the following csv format (first visualized as table, then as csv):

| Video_name                             | Second | Nr_larvae | Second | Nr_larvae | Second | Nr_larvae |
|----------------------------------------|--------|-----------|--------|-----------|--------|-----------|
| A1DF31_A1_2022-07-12-150920-0000_ffv1  | 0      | 12        | 32     | 11        | 44     | 10        |
| A1DF39_B1_2022-07-12-185350-0000_ffv1  | 0      | 11        | 52     | 10        |

As csv:
```
Video_name,Second,Nr_larvae,Second,Nr_larvae,Second,Nr_larvae
A1DF31_A1_2022-07-12-150920-0000_ffv1,0,12,32,11,44,10
A1DF39_B1_2022-07-12-185350-0000_ffv1,0,11,52,10
```
The tuple (Second, Nr_larvae) can be repeated as often as needed. Rows do not need to have the same number of columns.
If the target number of larvae is constant over the whole video, the file can be simplified specifying the
Second 0 and the target number of larvae once, so that the .csv file has 3 columns in total.

## About the Algorithm
The optimization algorithm is based on the Optuna framework, which is commonly used for hyper-parameter optimization
for neural networks. It suggests a set of hyper-parameters, which are then used to run the tracker on the videos.
Testing one set of hyper-parameters is called a trial.
Based on results from previous trials, Optuna will suggest new hyper-parameters to test.
Additionally, Optuna has an option to prune unpromising trials, which can be activated by setting the flag "--prune".
