import torch
import os
import pickle
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.append("dunl-compneuro\src")

import model


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default= [f"dopamine/results/dopamine_photometry_Day{i}_numwindow1_roi8_kernellength50_1kernels_1000unroll_2025_07_14_18_11_32" for i in range(1,4)] 
        #[f"dopamine/results/dopamine_photometry_numwindow1_neuron{i}_kernellength30_1kernels_1000unroll_2025_02_12_18_02_52" for i in range(19)]  
        #"dopamine/results/dopamine_photometry_numwindow1_neuron12_kernellength30_1kernels_1000unroll_2025_02_12_18_02_52"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="save path",
        default= "dopamine/results"
    )
    parser.add_argument(
        "--color-list",
        type=list,
        help="color list",
        default=[
            "black",
        ], 
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(8, 4),
    )

    args = parser.parse_args()
    params = vars(args)

    return params


def plot_code_histograms(
    code,
    params,
    out_path_name,
    idx
):

    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": 15,
            "axes.titlesize": 20,
            "legend.fontsize": 10,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
            "font.family": "sans-serif",
        }
    )

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=params["figsize"], sharex=True)

    # Plot histogram for Code 0
    axes.hist(
        code,
        bins=30,
        alpha=0.7,
        color=params["color_list"][0],
        edgecolor="black",
    )
    axes.set_title(f"Histogram of Code for ROI {idx}")
    axes.set_ylabel("Frequency")
    axes.grid(True)

    plt.tight_layout()
    #plt.savefig(out_path_name, bbox_inches="tight", pad_inches=0.02)
    #plt.close()
    plt.show()


def main():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is", device)

    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params_init = init_params()
    
    for idx, res_path in enumerate(params_init["res_path"]):
        print(res_path)

        # take parameters from the result path
        params = pickle.load(
            open(os.path.join(res_path, "params.pickle"), "rb")
        )
        for key in params_init.keys():
            params[key] = params_init[key]

        # create folders -------------------------------------------------------#
        model_path = os.path.join(
            res_path,
            "model",
            "model_final.pt",
        )

        out_path = os.path.join(
            res_path,
            "figures",
        )
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        postprocess_path = os.path.join(
            res_path,
            "postprocess",
        )
        
        if params["data_path"] == "":
            data_folder = params["data_folder"]
            filename_list = os.listdir(data_folder)
            data_path_list  = [
                f"{data_folder}/{x}" for x in filename_list if "trainready.pt" in x
            ]
        else:
            data_path_list = params["data_path"]
            
        
        # load codes ------------------------------------------------------#
        
        net = torch.load(model_path, map_location=device, weights_only=False)
        net.to(device)
        net.eval()

        for data_path in data_path_list:
            datafile_name = data_path.split("/")[-1].split(".pt")[0]

            xhat = torch.load(
                os.path.join(postprocess_path, "xhat_{}.pt".format(datafile_name))
            )

            y = torch.load(
                os.path.join(postprocess_path, "y_{}.pt".format(datafile_name))
            )    
            
            
        codes = xhat[0, idx, :].cpu().numpy()[0]

        plot_code_histograms(codes, params, "", idx)
        

if __name__ == "__main__":
    main()
