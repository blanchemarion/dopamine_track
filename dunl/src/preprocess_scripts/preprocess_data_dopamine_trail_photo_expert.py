import torch
import numpy as np
import argparse
import pickle
import os

def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-path",
        type=str,
        help="data path",
        default="dopamine/Data/Expert_mice/all_processed_Eva.pkl", 
    )
    parser.add_argument(
        "--kernel-num",
        type=int,
        help="number of convolutional kernels",
        default=1,
    )
    args = parser.parse_args()
    params = vars(args)

    return params

def main():
    params = init_params()

    print("data {} is being processed!".format(params["data_path"]))

    with open(params["data_path"],"rb") as f:
        data= pickle.load(f)

    for key, entry in data.items():
        if key == 'Day1':
            continue
        print(f"Processing entry: {key}")
        
        roi_indices = [8, 9, 13, 14, 15, 16]
    
        corrected_data = entry['corrected_data_iso'] # Shape: (num_rois, length_signal)
        corrected_data = corrected_data[:, roi_indices]
    
        num_rois = corrected_data.shape[1]
        trial_length = corrected_data.shape[0]
        num_trials = 1

        trials_all = []
        y = np.zeros((num_trials, num_rois, trial_length))
        baseline = np.zeros((num_trials, num_rois, 1))

        # Fill the structured array with the signals for each ROI
        for i in range(num_rois):
            y[0, i, :] =  corrected_data[:,i]
            baseline[0, i, 0] = 0
            trial_data = {
                "type": 0,
                "event0_onsets": [0]
            }
            trials_all.append(trial_data)

        print("Final shape of y:", y.shape)
        print("Total elements in y:", y.size)

        data_dict = {
            'y': y,
            'a': baseline,
            'kernel_num' : params["kernel_num"]
        }

        for idx, trial_info in enumerate(trials_all):
            data_dict[f'trial{idx}'] = trial_info

        save_path = f'dopamine\Data\Expert_mice\general_format_processed_{key}.npy'
        torch.save(data_dict, save_path)
        print(f"Processed data saved to {save_path}.")

        print("Data structure:")
        print(" - y: Dopamine response, shape (num_trials, num_rois, trial_length)")
        print(" - a: Baseline activity, shape (num_trials, num_rois, 1)")
        print(" - key kernel_num: number of kernels.")
        print(" - trial#: Metadata for each trial (event_onsets, type)")


if __name__ == '__main__':
    main()
