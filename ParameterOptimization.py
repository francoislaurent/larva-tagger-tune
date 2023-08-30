import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import optuna
import datetime
import subprocess
import argparse
import joblib


# ================================
# Example Usage (insert space whenever there is a new line):
# python Hyperparameter_Optimization_V3_cli.py
# WF-NTP '/Users/lillymay/Documents'
# '/Users/lillymay/OneDrive/Dokumente/02_Arbeit/Masson_Lab/Larvae_Tracking_Benchmarking/Data/ffv1' A1_2022-07-12-150920-0000_ffv1.avi,A2_2022-07-07-153918-0000_ffv1.avi
# 12,20
# ===============================================

def list_of_strings(arg):
    return arg.split(',')

def list_of_ints(arg):
    return [int(x) for x in arg.split(',')]

def csv_file_to_dict(arg):
    target_nr_dict = {}
    with open(arg, 'r') as file:
        lines = file.readlines()[1:]  # Skip the first line
        for line in lines:
            if ',' not in line:
                line = line.replace(';', ',')  # Adation to German csv file #TODO: make this more general
            row = line.strip().split(',')
            # Remove all '' from the list
            row = [x for x in row if x != '']

            target_nr_dict[row[0]] = []
            for i in range(1, len(row), 2):
                # Append a tuple of (time, target_nr) to the list
                target_nr_dict[row[0]].append((float(row[i]), int(row[i + 1])))
    return target_nr_dict


parser = argparse.ArgumentParser()

# Positional arguments
parser.add_argument('tracker', help='Tracker that should be used. Select between MWT, TIERPSY, and WF-NTP')
parser.add_argument('working_dir', help='Working directory is expected to have a subdirectory named '
                                        'TRACKER-cli for the tracker that is supposed to be used')
parser.add_argument('video_dir', help='Path to the directory in which all video files lie in')
parser.add_argument('--video_names', type=list_of_strings,
                    help='File name (including extension) of each video to include in the optimization, seperated by a '
                         'comma. Use "ALL" to include all videos in the specified folder.')  # TODO: Add option 'all'

# Optional arguments
parser.add_argument('--target_larvae_nr_list', type=list_of_ints,
                    help='Provide a list of the target number of larvae to detect in each video.'
                         ' Order must be according to the list provided in video_names.') # Todo: Delete this option, cause videos now also have option "ALL"
parser.add_argument('--target_larvae_nr_dict', type=csv_file_to_dict,  # TODO: rename this to csv not dict
                    help='Provide a csv file with the target number of larvae to detect in each video.')
parser.add_argument('--fps', type=int, default=30,
                    help='Frame rate of the videos.')
parser.add_argument('--downsampled_fps', type=int,
                    help='Provide a csv file with the target number of larvae to detect in each video.')  # TODO: check if default is required
parser.add_argument('--plot', action='store_true',
                    help='Plot the number of detected larvae over time for each hyperparameter set.')
parser.add_argument('--nr_trials', type=int, default=100)
parser.add_argument('--nr_processes', type=int, default=1,
                    help="Number of processes (each processing one video) that should run in parallel.")
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()


# Plot number of detected larvae over time
def plot_nr_detected_larvae(working_dir, date_time, dataframe, tracker, target_nr_larvae, video_id,
                            video_nr):
    plt.figure(figsize=(5,3))
    if tracker == 'MWT' or tracker == 'TIERPSY':
        dataframe.groupby('time').larva_id.nunique().plot()
    elif tracker == 'WF-NTP':
        dataframe.groupby('time').particle.nunique().plot()
    # Plot target number of larvae
    if type(target_nr_larvae) == list:
        plt.axhline(y=target_nr_larvae[video_nr], color='r', linestyle='--', label='Target Number')
    else:  # target_nr_larvae is a dict
        x, y = [], []
        for target_time, target_nr in target_nr_larvae[video_id]:
            x.append(target_time)
            y.append(target_nr)
        x.append(dataframe.time.max())
        y.append(target_nr_larvae[video_id][-1][1])
        plt.plot(x, y, color='r', linestyle='--',label='Target Number')
    plt.gca().set_ylim(bottom=0)
    plt.xlabel('Time in Seconds')
    plt.ylabel('Number of Detected Larvae')
    plt.title(video_id)
    plt.legend()
    save_path = os.path.join(working_dir, 'data', 'Optuna', video_id, date_time, f'nr_larvae_tracked_{video_id}.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def get_nr_detected_larvae_from_tracks(track_path, working_dir, date_time, tracker, target_nr_larvae,
                                       video_id, video_nr):  # parameters
    # track_path can be a spine file (for MWT and Tierpsy) or a track.p file (for WF-NTP)
    if tracker == 'MWT' or tracker == 'TIERPSY':
        spine_df = pd.read_csv(track_path, sep=' ', header=None)
        columns_points = []
        for i in range(1, 12):
            columns_points.extend([f'spinepoint{i}_x', f'spinepoint{i}_y'])
        spine_df.columns = ['date_time', 'larva_id', 'time'] + columns_points

        if args.plot:
            plot_nr_detected_larvae(working_dir, date_time, spine_df, tracker, target_nr_larvae, video_id, video_nr)
        return spine_df.groupby('time').larva_id.nunique()

    elif tracker == 'WF-NTP':
        # Save content of track.p file (WF-NTP output) in a pandas dataframe
        df = pd.read_pickle(track_path)
        df.reset_index(drop=True, inplace=True)
        # Add time column
        df['time'] = df['frame'] / args.fps

        if args.plot:
            plot_nr_detected_larvae(working_dir, date_time, df, tracker, target_nr_larvae, video_id, video_nr)
        return df.groupby('time').particle.nunique()


def get_hyperparameters(trial, tracker):
    date_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if tracker == 'MWT':
        pixel_thr1 = trial.suggest_int('pixel-thr1', 100, 255 - 1)
        pixel_thr2 = trial.suggest_int('pixel-thr2', pixel_thr1 + 1, 255)
        size_thr1 = trial.suggest_int('size-thr1', 1, 100)
        size_thr2 = trial.suggest_int('size-thr2', 1, 100)

        hyperparams = f'--frame-rate {args.fps} --pixel-thresholds {pixel_thr1} {pixel_thr2} --size-thresholds {size_thr1} {size_thr2} --pixel-size 0.073 --date-time {date_time}'.split(
            ' ')
    elif tracker == 'TIERPSY':
        # Only the parameters part of the GUI to set the parameters are included here
        mask_min_area = trial.suggest_int('mask-min-area', 0, 100)  # 1, 50
        mask_max_area = trial.suggest_int('mask-max-area', mask_min_area, 10000)  # 1e8
        thresh_C = trial.suggest_int('thresh-C', 0, 100)  # 10,20
        thresh_block_size = trial.suggest_int('thresh-block-size', 50, 70)  # 0, 500
        dilation_size = trial.suggest_int('dilation-size', 5, 15)  # 1, 100
        strel_size = trial.suggest_int('strel-size', 4, 8)
        worm_bw_thresh_factor = trial.suggest_float('worm-bw-thresh-factor', 0.8, 1.5)

        hyperparams = (f'--frame-rate {args.fps} --mask-min-area {mask_min_area} --mask-max-area {mask_max_area} '
                       f'--strel-size {strel_size} --worm-bw-thresh-factor {worm_bw_thresh_factor} '
                       f'--thresh-block-size {thresh_block_size} --dilation-size {dilation_size} --thresh-C {thresh_C} '
                       f'--pixel-size 0.073 --date-time {date_time}').split(' ')

    elif tracker == 'WF-NTP':
        # The hyperparameters that appear in the paper in Figure 11 are included here
        threshold = trial.suggest_int('threshold', 1, 20)
        opening = trial.suggest_int('opening', 1, 5)
        closing = trial.suggest_int('closing', 1, 5)
        min_size = trial.suggest_int('min-size', 10, 60)
        max_size = trial.suggest_int('max-size', min_size, 3000)
        minimum_ecc = trial.suggest_float('minimum-ecc', 0.5, 1.0)

        hyperparams = (f'--fps {args.fps} --px_to_mm 0.073 --threshold {threshold} --opening {opening} '
                       f'--closing {closing} --min_size {min_size} --max_size {max_size} --minimum_ecc {minimum_ecc} '
                       f'--skeletonize True --do_full_prune True').split(' ')

        if args.downsampled_fps is not None:
            print(f'Downsampling videos to {args.downsampled_fps} fps')
            hyperparams.extend(f'--downsampled_fps {args.downsampled_fps}'.split(' '))
            args.fps = args.downsampled_fps
    else:
        raise ValueError('Tracker not supported')
    return hyperparams, date_time


def analyze_one_video(video_path, hyperparams, working_dir, date_time, video_nr):
    video_id = os.path.basename(video_path).split('.')[0]
    print('Analyzing video: ', video_id)

    # For all trackers, the output directory is of the form: data/Optuna/VIDEO_ID/DATE_TIME
    if args.tracker == 'MWT':
        command = ['julia', '--project=.', f'src/{args.tracker.lower()}-cli.jl', video_path,
                   f'./data/Optuna/{video_id}']
    elif args.tracker == 'TIERPSY':
        command = ['julia', '--project=.', f'src/{args.tracker.lower()}-cli.jl', video_path,
                   f'./data/Optuna/{video_id}/{date_time}']
    elif args.tracker == 'WF-NTP':
        os.makedirs(f'{working_dir}/data/Optuna/{video_id}/{date_time}')
        command = ['python', f'{working_dir}/src/wf_ntp_cli.py', video_path, f'./data/Optuna/{video_id}/{date_time}']
    command.extend(hyperparams)

    try: # Bad Hyperparameter sets can cause errors, so we need to catch them and prune the trial
        if args.debug:
            out, err = subprocess.PIPE, subprocess.PIPE
        else:
            out, err = subprocess.DEVNULL, subprocess.DEVNULL

        # Run tracker for one video from working directory
        subprocess.run(command, cwd=working_dir, stdout=out, stderr=err)

        # Each tracker has a different name for the file that contains the tracks, so we need to handle this
        if args.tracker == 'MWT':
            spine_file_name = '20.spine'
        elif args.tracker == 'TIERPSY':
            spine_file_name = video_path.split('\\')[-1].split('.')[0] + '.spine'
        elif args.tracker == 'WF-NTP':
            spine_file_name = f'{video_id}_downsampled_track.p' if args.downsampled_fps != -1 else f'{video_id}_track.p'

        target_larvae_nr = args.target_larvae_nr_list if args.target_larvae_nr_list != None else args.target_larvae_nr_dict

        nr_larvae_tracked = get_nr_detected_larvae_from_tracks(
            f'{working_dir}/data/Optuna/{video_id}/{date_time}/{spine_file_name}', working_dir, date_time, args.tracker,
            target_larvae_nr, video_id, video_nr)

    except:
        # Prune optuna trial, because the hyperparameters caused an error in the tracking
        print('Exception when using hyperparameters: ', hyperparams, '\nTrying next set of hyperparameters...')
        raise optuna.TrialPruned()

    # calculate error for this video
    video_error = calculate_error(nr_larvae_tracked, target_larvae_nr, video_id, video_nr)
    return video_error


def calculate_error(nr_larvae_tracked, target_nr_larvae, video_id, video_nr):
    if type(target_nr_larvae) == list:
        target_nr = target_nr_larvae[video_nr]
        avrg_larvae_tracked = nr_larvae_tracked.mean()
        return abs(target_nr - avrg_larvae_tracked) / target_nr
    else:  # target_nr_larvae is a dict
        target_nr_list = target_nr_larvae[video_id]
        cumulative_error = 0.0
        current_target_nr = np.Inf
        for time, detected_larvae in nr_larvae_tracked.items():  # TODO: Control this
            for target_time, target_nr in target_nr_list:
                if time >= target_time:
                    current_target_nr = target_nr
                elif time < target_time:
                    break
            cumulative_error += abs(detected_larvae - current_target_nr) / current_target_nr
        return cumulative_error / len(nr_larvae_tracked)


def objective(trial):
    hyperparams, date_time = get_hyperparameters(trial, args.tracker)

    video_paths = [os.path.join(args.video_dir, video_name) for video_name in list(args.video_names)]
    working_dir = f'{args.working_dir}/{args.tracker.lower()}-cli'

    total_error = 0.0
    parallel_run_params = [(current_video_path, hyperparams, working_dir, date_time, video_nr) for
                           video_nr, current_video_path in enumerate(video_paths)]

    # Run tracker on videos in parallel, using the number or parallel processes specified in args.nr_processes
    with multiprocessing.Pool(processes=args.nr_processes) as pool: #len(video_paths)
        errors = pool.starmap(analyze_one_video, parallel_run_params)

    for video_error in errors:
        total_error += video_error

        # trial.report(video_error, video_nr)

        # Handle pruning based on the intermediate value.
        # if trial.should_prune():
        #   raise optuna.TrialPruned()

    return total_error / len(video_paths)


def main():
    # Check if provided tracker is supported
    if args.tracker != 'MWT' and args.tracker != 'TIERPSY' and args.tracker != 'WF-NTP':
        raise ValueError('Tracker not supported')

    # Check if target number of larvae is provided correctly
    if args.target_larvae_nr_list is None and args.target_larvae_nr_dict is None \
            or args.target_larvae_nr_list is not None and args.target_larvae_nr_dict is not None:
        raise ValueError('Target number of larvae must be provided EITHER as a list (--target_larvae_nr_list) '
                         'or as a csv file (--target_larvae_nr_dict).')

    nr_target_values = len(args.target_larvae_nr_list) if args.target_larvae_nr_list != None \
        else len(args.target_larvae_nr_dict.keys())


    if len(args.video_names) != nr_target_values:
        raise ValueError('Number of videos and number of provided target larvae numbers must be the same.')
    print(f'Number of videos included in optimization: {len(args.video_names)}')

    if not os.path.exists(f'{args.working_dir}/{args.tracker.lower()}-cli/data/Optuna'):
        os.mkdir(f'{args.working_dir}/{args.tracker.lower()}-cli/data/Optuna')

    # Create optuna study and optimize hyperparameters
    study = optuna.create_study(direction='minimize', study_name=f'Parameter_Optimization_{args.tracker}')
    study.optimize(objective, n_trials=args.nr_trials)

    # Save study and results
    date_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    joblib.dump(study, f'{args.working_dir}/{args.tracker.lower()}-cli/data/Optuna/{study.study_name}_{date_time}.pkl')
    df = study.trials_dataframe()
    df.to_csv(f'{args.working_dir}/{args.tracker.lower()}-cli/data/Optuna/{study.study_name}_{date_time}.csv')

if __name__ == "__main__":
    main()
