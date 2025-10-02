import torch
import os
import pickle
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sys

sys.path.append("dunl-compneuro\src")
sys.path.append("")

import model

def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default= "dopamine/results/dopamine_photometry_Day1_numwindow1_roi13_kernellength50_1kernels_1000unroll_2025_07_14_20_49_49"  
        #"dopamine/results/dopamine_photometry_numwindow1_roi16_kernellength30_1kernels_1000unroll_2025_02_20_15_49_52"
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        help="sampling rate",
        default=20,
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(8, 2),
    )

    args = parser.parse_args()
    params = vars(args)

    return params



def plot_y_yhat_xhat(y, yhat, xhat):

    print(y.shape)
    print(yhat.shape)
    print(xhat.shape)

    i = 0
    j = 13

    yi = y[i, j, :].clone().detach().cpu().numpy()
    yihat = yhat[i, j, :].clone().detach().cpu().numpy()[0]
    codehat = xhat[i, j, :].clone().detach().cpu().numpy()[0]
    codehat = np.pad(codehat, (49, 0), mode='constant', constant_values=0)
    
    print(yi.shape)
    print(yihat.shape)
    print(codehat.shape)

    axes_fontsize = 10
    legend_fontsize = 8
    tick_fontsize = 10
    title_fontsize = 10
    fontfamily = "sans-serif"

    # upadte plot parameters
    # style
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "axes.labelsize": axes_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "text.latex.preamble": r"\usepackage{bm}",
            "axes.unicode_minus": False,
            "font.family": fontfamily,
        }
    )

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.plot(yi, color="black", label="raw", lw=0.7)
    plt.plot(yihat, color="blue", label="rec", lw=0.7)
    plt.plot(codehat, ".", color="green", alpha=0.7, lw=0.5, label="code")
    plt.xlabel("Time")
    plt.legend()
    plt.title(f'ROI {j}')
    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.show()
    plt.close()
    

def main():
    print("Predict.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is", device)

    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params_init = init_params()

    # take parameters from the result path
    params = pickle.load(
        open(os.path.join(params_init["res_path"], "params.pickle"), "rb")
    )
    for key in params_init.keys():
        params[key] = params_init[key]

    # create folders -------------------------------------------------------#
    model_path = os.path.join(
        params["res_path"],
        "model",
        "model_final.pt",
    )

    out_path = os.path.join(
        params["res_path"],
        "figures",
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    postprocess_path = os.path.join(
        params["res_path"],
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

    """for data_path in data_path_list:
        datafile_name = data_path.split("/")[-1].split(".pt")[0]

        xhat = torch.load(
            os.path.join(postprocess_path, "xhat_{}.pt".format(datafile_name))
        )
        yhat = torch.load(
            os.path.join(postprocess_path, "yhat_{}.pt".format(datafile_name))
        )
        y = torch.load(
            os.path.join(postprocess_path, "y_{}.pt".format(datafile_name))
        )"""
        
    xhat = torch.load(
        os.path.join(postprocess_path, "xhat.pt")
    )
    yhat = torch.load(
        os.path.join(postprocess_path, "yhat.pt")
    )
    y = torch.load(
        os.path.join(postprocess_path, "y.pt")
    )

    plot_y_yhat_xhat(y, yhat, xhat)


if __name__ == "__main__":
    main()

