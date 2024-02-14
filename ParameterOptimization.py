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


def list_of_strings(arg):
    return arg.split(',')


def list_of_ints(arg):
    return [int(x) for x in arg.split(',')]


# Converts the csv file with the target number of larvae to a dictionary
def csv_file_to_dict(arg):
    target_nr_dict = {}
    with open(arg, 'r') as file:
        lines = file.readlines()[1:]  # Skip the first line
        for line in lines:
            # Replace ; with , to make it compatible with the German csv file #TODO: make this more general
            if ',' not in line:
                line = line.replace(';', ',')

            row = line.strip().split(',')

            # Remove all '' from the list, which occur because of different row lengths
            row = [x for x in row if x != '']

            # Create a list for each video, which contains tuples of (time, target_nr) values
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
parser.add_argument('target_larvae_nr',
                    help='Provide a csv file with the target number of larvae to detect in each video.')

# Optional arguments
parser.add_argument('--video_names',
                    help='If you do not want to use all videos in the provided video directory, you can specify a list '
                         'of video names (including extension) that should be used for the optimization.')
parser.add_argument('--fps', type=int, default=30,
                    help='Frame rate of the videos.')
parser.add_argument('--downsampled_fps', type=int,
                    help='For WF-NTP, there is the option to downsample the '
                         'videos to a lower frame rate to speed up processing.')
parser.add_argument('--plot', action='store_true',
                    help='Plot the number of detected larvae over time for each hyperparameter set.')
parser.add_argument('--nr_trials', type=int, default=100)
parser.add_argument('--nr_processes', type=int, default=1,
                    help="Number of processes (each processing one video) that should run in parallel.")
parser.add_argument('--debug', action='store_true', help='Print debug information.')
parser.add_argument('--prune', action='store_true',
                    help='Useful if you have more videos than processes, as it will prune unpromising trials after '
                         'processing the first video.')
parser.add_argument('--output_dir', help='Directory for Optuna study file.')
parser.add_argument('--static-args', help='Constant arguments to the tracker (pass them quoted).', default='')

args = parser.parse_args()

args.target_larvae_nr = csv_file_to_dict(args.target_larvae_nr)
args.video_names = list_of_strings(args.video_names)


# Plot number of detected larvae over time
def plot_nr_detected_larvae(output_dir, date_time, dataframe, tracker, video_id):
    plt.figure(figsize=(5, 3))

    # Get number of detected larvae for each time point
    if tracker == 'MWT' or tracker == 'TIERPSY':
        dataframe.groupby('time').larva_id.nunique().plot()
    elif tracker == 'WF-NTP':
        dataframe.groupby('time').particle.nunique().plot()

    # Plot target number of larvae
    x, y = [], []
    for target_time, target_nr in args.target_larvae_nr[video_id]:
        x.append(target_time)
        y.append(target_nr)
    x.append(dataframe.time.max())
    y.append(args.target_larvae_nr[video_id][-1][1])
    plt.plot(x, y, color='r', linestyle='--', label='Target Number')

    # Format plot
    plt.gca().set_ylim(bottom=0)
    plt.xlabel('Time in Seconds')
    plt.ylabel('Number of Detected Larvae')
    plt.title(video_id)
    plt.legend()

    # Save plot
    save_path = os.path.join(output_dir, video_id, date_time, f'nr_larvae_tracked_{video_id}.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


# Processed the different output files of the trackers to get the number of detected larvae for each time point
def get_nr_detected_larvae_from_tracks(track_path, output_dir, date_time, tracker, video_id, video_nr):
    # track_path can be a spine file (for MWT and Tierpsy) or a track.p file (for WF-NTP)
    if tracker == 'MWT' or tracker == 'TIERPSY':
        # Save content of spine file (MWT and Tierpsy output) in a pandas dataframe
        spine_df = pd.read_csv(track_path, sep=' ', header=None)
        columns_points = []
        for i in range(1, 12):
            columns_points.extend([f'spinepoint{i}_x', f'spinepoint{i}_y'])
        spine_df.columns = ['date_time', 'larva_id', 'time'] + columns_points

        if args.plot:
            plot_nr_detected_larvae(output_dir, date_time, spine_df, tracker, video_id)
        return spine_df.groupby('time').larva_id.nunique()

    elif tracker == 'WF-NTP':
        # Save content of track.p file (WF-NTP output) in a pandas dataframe
        df = pd.read_pickle(track_path)
        df.reset_index(drop=True, inplace=True)
        # Add time column
        df['time'] = df['frame'] / args.fps

        if args.plot:
            plot_nr_detected_larvae(output_dir, date_time, df, tracker, video_id)
        return df.groupby('time').particle.nunique()


# Get hyperparameters for the tracker that should be optimized using optuna
def get_hyperparameters(trial, tracker):
    date_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if tracker == 'MWT':
        pixel_thr1 = trial.suggest_int('pixel-thr1', 100, 255 - 1)
        pixel_thr2 = trial.suggest_int('pixel-thr2', pixel_thr1 + 1, 255)
        size_thr1 = trial.suggest_int('size-thr1', 1, 100)
        size_thr2 = trial.suggest_int('size-thr2', 1, 100)

        hyperparams = (f'--frame-rate {args.fps} --pixel-thresholds {pixel_thr1} {pixel_thr2} '
                       f'--size-thresholds {size_thr1} {size_thr2} --pixel-size 0.073 '
                       f'--date-time {date_time}').split(' ')
    elif tracker == 'TIERPSY':
        mask_min_area = trial.suggest_int('mask-min-area', 0, 100)  # 1, 50
        mask_max_area = trial.suggest_int('mask-max-area', mask_min_area, 10000)  # 1e8
        thresh_C = trial.suggest_int('thresh-C', 0, 100)  # 10,20
        thresh_block_size = trial.suggest_int('thresh-block-size', 50, 70)  # 0, 500
        dilation_size = trial.suggest_int('dilation-size', 5, 15)  # 1, 100
        strel_size = trial.suggest_int('strel-size', 4, 8)
        worm_bw_thresh_factor = trial.suggest_float('worm-bw-thresh-factor', 0.8, 1.5)

        hyperparams = (f'--frame-rate {args.fps} --mask-min-area {mask_min_area} --mask-max-area {mask_max_area} '
                       f'--strel-size {strel_size} --worm-bw-thresh-factor {worm_bw_thresh_factor} '
                       f'--thresh-block-size {thresh_block_size} --dilation-size {dilation_size} '
                       f'--thresh-C {thresh_C} --pixel-size 0.073 --date-time {date_time}').split(' ')

    elif tracker == 'WF-NTP':
        # The hyperparameters that appear in the WF-NTP paper in Figure 11 are included here
        threshold = trial.suggest_int('threshold', 1, 20)
        opening = trial.suggest_int('opening', 1, 5)
        closing = trial.suggest_int('closing', 1, 5)
        min_size = trial.suggest_int('min-size', 10, 60)
        max_size = trial.suggest_int('max-size', min_size, 3000)
        minimum_ecc = trial.suggest_float('minimum-ecc', 0.5, 1.0)

        hyperparams = (f'--fps {args.fps} --px_to_mm 0.073 --threshold {threshold} --opening {opening} '
                       f'--closing {closing} --min_size {min_size} --max_size {max_size} --minimum_ecc {minimum_ecc} '
                       f'--skeletonize True --do_full_prune True').split(' ')

        # Add downsampling option if specified via command line
        if args.downsampled_fps is not None:
            print(f'Downsampling videos to {args.downsampled_fps} fps')
            hyperparams.extend(f'--downsampled_fps {args.downsampled_fps}'.split(' '))

            # Adjust frame rate for WF-NTP for correct downstream processing
            args.fps = args.downsampled_fps

    if args.static_args:
        hyperparams.extend(args.static_args.split(' '))
    return hyperparams, date_time


def analyze_one_video(video_path, hyperparams, working_dir, output_dir, date_time, video_nr, trial):
    video_id = os.path.basename(video_path).split('.')[0]
    print('Analyzing video: ', video_id)

    # If output_dir is not defined, for all trackers, the output directory is of the form:
    #   data/Optuna/VIDEO_ID/DATE_TIME
    # Variable output_dir refers to the data/Optuna part.
    # If output_dir is defined, it should be an absolute path like the other directories.
    if output_dir is None:
        output_dir = f'./data/Optuna'

    # Get the basic command for the specified tracker
    if args.tracker == 'MWT':
        command = ['julia', '--project=.', f'src/{args.tracker.lower()}-cli.jl', video_path,
                   f'{output_dir}/{video_id}']
    elif args.tracker == 'TIERPSY':
        command = ['julia', '--project=.', f'src/{args.tracker.lower()}-cli.jl', video_path,
                   f'{output_dir}/{video_id}/{date_time}']
    elif args.tracker == 'WF-NTP':
        os.makedirs(f'{output_dir}/{video_id}/{date_time}')
        command = ['python', f'{working_dir}/src/wf_ntp_cli.py', video_path, f'{output_dir}/{video_id}/{date_time}']
    command.extend(hyperparams)

    try:  # Bad Hyperparameter sets can cause errors, so we need to catch them and prune the trial
        if args.debug:
            out, err = subprocess.PIPE, subprocess.PIPE
        else:
            out, err = subprocess.DEVNULL, subprocess.DEVNULL

        # Run tracker for one video from working directory
        ret = subprocess.run(command, cwd=working_dir, stdout=out, stderr=err)

        if args.debug and ret.stderr:
            print(ret.stderr.decode('utf8'))

        # Each tracker has a different name for the file that contains the tracks, so we need to handle this
        if args.tracker == 'MWT':
            spine_file_name = '20.spine'
        elif args.tracker == 'TIERPSY':
            spine_file_name = video_path.split('\\')[-1].split('.')[0] + '.spine'
        elif args.tracker == 'WF-NTP':
            spine_file_name = f'{video_id}_downsampled_track.p' if args.downsampled_fps != -1 else f'{video_id}_track.p'

        # Get number of detected larvae from output tracks
        if not os.path.isabs(output_dir):
            output_dir = f'{working_dir}/{output_dir}'
        nr_larvae_tracked = get_nr_detected_larvae_from_tracks(
            f'{output_dir}/{video_id}/{date_time}/{spine_file_name}', output_dir, date_time, args.tracker,
            video_id, video_nr)

    except:
        # Prune optuna trial, because the hyperparameters caused an error in the tracking
        print('Exception when using hyperparameters: ', hyperparams, '\nTrying next set of hyperparameters...')
        raise optuna.TrialPruned()

    # Calculate error for this video
    video_error = calculate_error(nr_larvae_tracked, video_id)

    # Handle pruning if specified via command line
    if args.prune:
        # Report intermedidate error
        trial.report(video_error, video_nr)

        # The current trial is pruned if the error is too high
        if trial.should_prune():
            print('Pruning trial because of high error for one video: ', video_error)
            raise optuna.TrialPruned()
    return video_error


# Calculate the mean deviation of the detected number of larvae from the target number of larvae
def calculate_error(nr_larvae_tracked, video_id):
    # Get target number of larvae for this video
    target_nr_list = args.target_larvae_nr[video_id]

    cumulative_error = 0.0
    current_target_nr = np.Inf
    # Calculate cumulative error over the entire video
    for time, detected_larvae in nr_larvae_tracked.items():
        # For one time point in the output track file, get the target number of larvae
        for target_time, target_nr in target_nr_list:
            if time >= target_time:
                current_target_nr = target_nr
            elif time < target_time:
                break

        # Add the error for this time point (the deviation of the detected number of larvae from the actual number)
        # to the cumulative error
        cumulative_error += abs(detected_larvae - current_target_nr) / current_target_nr

    # Average the cumulative error over the entire video to get the mean error
    return cumulative_error / len(nr_larvae_tracked)


# Objective function for optuna
def objective(trial):
    # Get hyperparameters for this trial
    hyperparams, date_time = get_hyperparameters(trial, args.tracker)

    video_paths = [os.path.join(args.video_dir, video_name) for video_name in list(args.video_names)]
    working_dir = f'{args.working_dir}/{args.tracker.lower()}-cli'

    total_error = 0.0
    # Create a list of running parameters for each video so that they can be run in parallel
    parallel_run_params = [(current_video_path, hyperparams, working_dir, args.output_dir, date_time, video_nr, trial) for
                           video_nr, current_video_path in enumerate(video_paths)]

    # Run tracker on videos in parallel, using the number or parallel processes specified in args.nr_processes
    with multiprocessing.Pool(processes=args.nr_processes) as pool:
        # Each process returns the error for one video
        errors = pool.starmap(analyze_one_video, parallel_run_params)

    # Calculate the cumulative error over all videos
    for video_error in errors:
        total_error += video_error

    # Return the mean error over all videos, representing the mean error for that set of hyperparameters
    return total_error / len(video_paths)


def main():
    # Check if provided tracker is supported
    if args.tracker != 'MWT' and args.tracker != 'TIERPSY' and args.tracker != 'WF-NTP':
        raise ValueError('Tracker not supported')

    # Obtain video names from video directory if no video names are provided
    if args.video_names is None:
        args.video_names = os.listdir(args.video_dir)

    # If necessary, filter video names to only include videos that are used for optimization
    # and make sure no video is missing
    if len(args.video_names) != len(args.target_larvae_nr.keys()):
        for video_name in args.video_names:
            if video_name.replace('.avi', '') not in args.target_larvae_nr.keys():
                raise ValueError(
                    f'Video name {video_name.replace(".avi", "")} not found in target number of larvae dictionary.')
        # Filter dictionary to only include videos that are used for optimization
        args.target_larvae_nr = {k: v for k, v in args.target_larvae_nr.items() if f'{k}.avi' in args.video_names}

    print(f'Number of videos included in optimization: {len(args.video_names)}')

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f'{args.working_dir}/{args.tracker.lower()}-cli/data/Optuna'
    os.makedirs(output_dir)

    # Create optuna study and optimize hyperparameters
    study = optuna.create_study(direction='minimize', study_name=f'Parameter_Optimization_{args.tracker}')
    study.optimize(objective, n_trials=args.nr_trials)

    # Save study and results
    date_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    joblib.dump(study, f'{output_dir}/{study.study_name}_{date_time}.pkl')
    df = study.trials_dataframe()
    df.to_csv(f'{output_dir}/{study.study_name}_{date_time}.csv')


if __name__ == "__main__":
    main()
