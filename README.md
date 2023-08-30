# larva-tagger-tune

## Project Description
This project is part of the Larva Tagger Project, aiming to monitor the behavior of Drosophila Larvae in video
recordings. This repository contains the code for the autimated hyperparameter optimization of the Larva Tagger.
Currently, three Larva Taggers are supported:
- MWT (Multi-Worm Tracker, [Paper](https://doi.org/10.1038%2Fnmeth.1625), [Code](https://gitlab.com/larvataggerpipelines/mwt-cli))
- Tierpsy ([Paper](https://doi.org/10.1038%2Fs41592-018-0112-1), [Code](https://gitlab.com/larvataggerpipelines/tierpsy-cli))
- WF-NTP (Wide field-of-view Nematode Tracking Platform, [Paper](https://doi.org/10.1038/s41596-020-0321-9), [Code]())

## Installation


## Usage
An example:
```python ParameterOptimization.py MWT```

### Target Number File
The target number file is a .csv file containing the target number of worms for each video. The file has to be in the following format:
```csv```