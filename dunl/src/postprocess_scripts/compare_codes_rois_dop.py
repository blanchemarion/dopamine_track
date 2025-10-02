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

#list_rois=[0, 2, 4, 5, 9, 12, 13, 15, 16, 17]
list_rois=[2, 4, 13, 15, 16]


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default= [f"dopamine/results/dopamine_photometry_numwindow1_roi{i}_kernellength30_1kernels_1000unroll" for i in list_rois]
        #"dopamine/results/dopamine_photometry_numwindow1_neuron0_kernellength30_1kernels_1000unroll"
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



def plot_y_yhat_xhat(y, yhat, xhat, j):

    i = 0

    yi = y[i, j, :].clone().detach().cpu().numpy()
    yihat = yhat[i, j, :].clone().detach().cpu().numpy()[0]
    codehat = xhat[i, j, :].clone().detach().cpu().numpy()[0]

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



def plot_code_heatmap(xhat, list_rois, fs=20.0, cmap='plasma'):

    xhat_np = xhat.detach().cpu().numpy().squeeze()
    
    selected = xhat_np[list_rois, :]  
    masked = np.ma.masked_where(selected == 0, selected)

    n_rois, T = xhat_np.shape
    t = np.arange(T) / fs

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        masked,
        aspect='auto',
        origin='lower',
        extent=[t[0], t[-1], 0, n_rois],
        cmap=cmap,
        interpolation='none'
    )
    fig.colorbar(im, label='Code amplitude')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ROI index')
    plt.tight_layout()
    plt.show()
    
    

def plot_code_raster(xhat, list_rois, fs=20.0, cmap='viridis'):

    x = xhat.detach().cpu().numpy().squeeze()   # shape (19, T)
    T = x.shape[1]
    times = np.arange(T) / fs
    
    masked = np.ma.masked_where(x == 0, x)

    fig, ax = plt.subplots(figsize=(12, len(list_rois)*0.4))
    for row, roi in enumerate(list_rois):
        ev_idxs = np.nonzero(masked[roi])[0]
        if ev_idxs.size == 0:
            continue
        amps = masked[roi, ev_idxs]
        ax.scatter(
            ev_idxs/fs,                    # x-position
            np.full_like(ev_idxs, row),    # y-position
            c=amps,                        # color = amplitude
            cmap=cmap,
            s=20,                          # marker size
            vmin=masked[masked>0].min(), 
            vmax=masked.max(),
            edgecolors='none'
        )

    ax.set_yticks(np.arange(len(list_rois)))
    ax.set_yticklabels(list_rois)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ROI index")
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, label="Code amplitude")
    plt.tight_layout()
    plt.show()


def main():
    print("Predict.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is", device)

    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params_init = init_params()
    
    for idx, res_path in enumerate(params_init["res_path"]):
        
        print(idx)
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
            yhat = torch.load(
                os.path.join(postprocess_path, "yhat_{}.pt".format(datafile_name))
            )
            y = torch.load(
                os.path.join(postprocess_path, "y_{}.pt".format(datafile_name))
            )

        #plot_y_yhat_xhat(y, yhat, xhat, j=list_rois[idx])
        plot_code_raster(xhat, list_rois, fs=20.0, cmap='viridis')
        plot_code_heatmap(xhat, list_rois)

    ##

if __name__ == "__main__":
    main()

