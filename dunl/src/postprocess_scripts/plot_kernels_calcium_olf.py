import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt


import sys

sys.path.append("dunl-compneuro\src")

import model


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--res-path",
        type=str,
        help="res path",
        default= "sparseness/results/olfactorycalciumkernellength20num1_2025_02_20_11_30_36"
        #"sparseness/results/olfactorycalciumkernellength20num1_2025_02_18_15_32_24"
        #"sparsenessresults\calcium_unsupervised_numwindow1_neuron0_kernellength20_1kernels_1000unroll_2025_02_05_19_07_13"
        # "dopamine\Results\dopamine_photometry_numwindow1_neuron1_kernellength30_1kernels_1000unroll_2025_02_05_16_07_40",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        help="sampling rate",
        default=10,
    )
    parser.add_argument(
        "--color-list",
        type=list,
        help="color list",
        default=[
            "black",
            "orange",
            "blue",
            "red",
        ], 
    )
    parser.add_argument(
        "--figsize",
        type=tuple,
        help="figsize",
        default=(8, 8),
    )

    args = parser.parse_args()
    params = vars(args)

    return params


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

    # load model ------------------------------------------------------#
    net = torch.load(model_path, map_location=device, weights_only=False)
    net.to(device)
    net.eval()

    kernels = np.squeeze(net.get_param("H").clone().detach().cpu().numpy())

    plot_kernel(kernels, params, out_path)


def plot_kernel(kernels, params, out_path):
    axes_fontsize = 15
    legend_fontsize = 8
    tick_fontsize = 15
    title_fontsize = 20
    fontfamily = "sans-serif"

    # Update plot parameters
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

    # Determine the number of subplots dynamically
    kernel_num = params.get("kernel_num", len(kernels))
    fig, axn = plt.subplots(
        1, kernel_num, sharex=True, sharey=True, figsize=params["figsize"]
    )

    # Handle case where there is only one kernel
    if kernel_num == 1:
        axn = [axn]  # Make axn iterable for consistency

    for ax in axn:
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    t = np.linspace(
        0, params["kernel_length"] / params["sampling_rate"], params["kernel_length"]
    )

    print(f't is {t.shape}')
    print(kernels.shape)

    for ctr in range(kernel_num):
        ax = axn[ctr]
        ax.axhline(0, color="gray", lw=0.3)

        #ax.plot(t, kernels[ctr], color=params["color_list"][ctr])
        # uncomment next line if only one kernel
        ax.plot(t, kernels, color=params["color_list"][ctr])

        titles = params.get(
            "titles",
            [r"$\textbf{Kernel\ %d}$" % (i + 1) for i in range(kernel_num)],
        )
        ax.set_title(titles[ctr])

        xtic = np.array([0, 0.5, 1]) * params["kernel_length"] / params["sampling_rate"]
        ax.set_xticks(xtic)
        ax.set_xticklabels(xtic)

        if ctr == kernel_num // 2:
            ax.set_xlabel("Time [s]", labelpad=0)

    fig.tight_layout(pad=0.8, w_pad=0.7, h_pad=0.5)
    plt.savefig(
        os.path.join(out_path, "kernels.png"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()

    print(f"Plotting of kernels is done. Plots are saved at {out_path}")



if __name__ == "__main__":
    main()
