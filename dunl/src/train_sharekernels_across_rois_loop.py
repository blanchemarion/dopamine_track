import numpy as np
import torch
import configmypy
import os
import json
import pickle
import tensorboardX
from datetime import datetime
from tqdm import tqdm
import argparse

import model, lossfunc, boardfunc, datasetloader


def init_params():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-folder",
        type=str,
        help="config folder",
        default="dunl-compneuro/config"
    )
    parser.add_argument(
        "--config-filename",
        type=str,
        help="config filename",
        default="olfactory_calcium_config.yaml"
    )
    args = parser.parse_args()
    params = vars(args)
    return params


def main():
    print("Train DUNL on neural data.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is", device)

    # init parameters -------------------------------------------------------#
    print("init parameters.")
    params_init = init_params()
    pipe = configmypy.ConfigPipeline(
        [
            configmypy.YamlConfig(
                params_init["config_filename"],
                config_name="default",
                config_folder=params_init["config_folder"],
            ),
            configmypy.ArgparseConfig(
                infer_types=True, config_name=None, config_file=None
            ),
            configmypy.YamlConfig(config_folder=params_init["config_folder"]),
        ]
    )
    params = pipe.read_conf()
    params["config_folder"] = params_init["config_folder"]
    params["config_filename"] = params_init["config_filename"]

    if params["code_q_regularization_matrix_path"]:
        params["code_q_regularization_matrix"] = (
            torch.load(params["code_q_regularization_matrix_path"]).float().to(device)
        )

    if not params["share_kernels_among_neurons"]:
        raise NotImplementedError(
            "This script is for sharing kernels among neurons. Set share_kernels_among_neurons=True!"
        )

    print("Exp: {}".format(params["exp_name"]))

    # Get list of trainready files.
    if params["data_path"] == "":
        data_folder = params["data_folder"]
        filename_list = os.listdir(data_folder)
        data_path_list = [
            f"{data_folder}/{x}" for x in filename_list if "HW1_kernellength20_kernelnum1_trainready.pt" in x
        ]
    else:
        data_path_list = params["data_path"]

    print("There are {} dataset(s) in the folder.".format(len(data_path_list)))
    print(data_path_list)

    # Loop over each trainready file
    for data_path_cur in data_path_list:
        print("Training on file:", data_path_cur)
        # Create a dataset for the current file.
        dataset = datasetloader.DUNLdataset(data_path_cur)
        num_data = len(dataset)
        if params["train_val_split"] < 1:
            num_train = int(np.floor(num_data * params["train_val_split"]))
            num_val = num_data - num_train
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset,
                [num_train, num_val],
            )
        else:
            train_dataset = dataset

        # Create a DataLoader for the current dataset.
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=params["train_data_shuffle"],
            batch_size=params["train_batch_size"],
            num_workers=params["train_num_workers"],
        )

        # Create a unique output directory using the file name and current time.
        file_basename = data_path_cur.split("general_format_processed_")[-1].split("_kernellength")[0]
        random_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        out_path = os.path.join("sparseness/results", f"{params['exp_name']}_{file_basename}_{random_date}")
        params["out_path"] = out_path
        os.makedirs(os.path.join(out_path, "model"), exist_ok=True)
        os.makedirs(os.path.join(out_path, "figures"), exist_ok=True)

        # Dump parameters for this run.
        with open(os.path.join(out_path, "params.txt"), "w") as file:
            params_clone = params.copy()
            params_clone["code_q_regularization_matrix"] = str(params.get("code_q_regularization_matrix"))
            file.write(json.dumps(params_clone, sort_keys=True, separators=("\n", ":")))
        with open(os.path.join(out_path, "params.pickle"), "wb") as file:
            pickle.dump(params_clone, file)

        # Create board if enabled.
        if params["enable_board"]:
            writer = tensorboardX.SummaryWriter(os.path.join(out_path))
            writer.add_text("params", str(params))
            writer.flush()

        # Create model ---------------------------------------------------------#
        if params["kernel_initialization"]:
            kernel_init = torch.load(params["kernel_initialization"])
            if params["kernel_initialization_needs_adjustment_of_time_bin_resolution"]:
                bin_res = int(kernel_init.shape[-1] / params["kernel_length"])
                kernel_init = np.add.reduceat(
                    kernel_init,
                    np.arange(0, kernel_init.shape[-1], bin_res),
                    axis=-1,
                )
        else:
            kernel_init = None

        print("Creating model for file", data_path_cur)
        net = model.DUNL1D(params, kernel_init)
        net.to(device)

        if params["kernel_nonneg"]:
            net.nonneg_kernel(params["kernel_nonneg_indicator"])
        if params["kernel_normalize"]:
            net.normalize_kernel()

        # Create optimizer and scheduler ---------------------------------------#
        print("Creating optimizer and scheduler for training.")
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=params["optimizer_lr"],
            eps=params["optimizer_adam_eps"],
            weight_decay=params["optimizer_adam_weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params["optimizer_lr_step"],
            gamma=params["optimizer_lr_decay"],
        )

        # Create loss criteria  ------------------------------------------------#
        criterion = lossfunc.DUNL1DLoss(params["model_distribution"])
        if params["kernel_smoother"]:
            criterion_kernel_smoother = lossfunc.Smoothloss()
        if params["code_l1loss_bp"]:
            criterion_l1_code = lossfunc.l1loss()

        # Training loop for the current file  ----------------------------------#
        print("Start training on", data_path_cur)
        for epoch in tqdm(range(params["train_num_epochs"]), disable=params["tqdm_prints_disable"]):
            net.train()
            total_train_loss_list = []
            total_train_loss_ae_list = []

            for idx, (y_load, x_load, a_load, type_load) in tqdm(enumerate(train_loader), disable=params["tqdm_prints_inside_disable"]):
                  
                # Collapse neurons into the trial dimension.
                y = torch.reshape(y_load, (int(y_load.shape[0] * y_load.shape[1]), 1, y_load.shape[2]))
                a = torch.reshape(a_load, (int(a_load.shape[0] * a_load.shape[1]), 1, a_load.shape[2]))
                # Repeat x for each neuron.
                x = torch.repeat_interleave(x_load, a_load.shape[1], dim=0)

                # Move data to device.
                y = y.to(device)
                x = x.to(device)
                a = a.to(device)

                x_code_supp = x if params["code_supp"] else None

                # Forward pass.
                xhat, a_est = net.encode(y, a, x_code_supp)
                yhat = net.decode(xhat, a_est)

                # Compute losses.
                loss_ae = criterion(y, yhat)
                loss_kernel_smoother = criterion_kernel_smoother(net.get_param("H")) if params["kernel_smoother"] else 0.0
                loss_l1_code = criterion_l1_code(xhat) if params["code_l1loss_bp"] else 0.0

                loss = (
                    loss_ae +
                    params["kernel_smoother_penalty_weight"] * loss_kernel_smoother +
                    params["code_l1loss_bp_penalty_weight"] * loss_l1_code
                )

                total_train_loss_list.append(loss.item())
                total_train_loss_ae_list.append(loss_ae.item())

                # Backward pass.
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # Kernel projections/normalization.
                if params["kernel_nonneg"]:
                    net.nonneg_kernel(params["kernel_nonneg_indicator"])
                if params["kernel_normalize"]:
                    net.normalize_kernel()

            scheduler.step()

            if (epoch + 1) % params["log_info_epoch_period"] == 0:
                total_train_loss = np.mean(total_train_loss_list)
                total_train_loss_ae = np.mean(total_train_loss_ae_list)
                print("Epoch {}: total_train_loss {:.4f}, total_train_loss_ae {:.4f}".format(
                    epoch, total_train_loss, total_train_loss_ae))
                if params["enable_board"]:
                    writer.add_scalar("loss/train", total_train_loss, epoch)
                    writer.add_scalar("loss/train_ae", total_train_loss_ae, epoch)
                    if params["kernel_smoother"]:
                        writer.add_scalar("loss/kernel_smoother", loss_kernel_smoother.item(), epoch)
                    writer.flush()

            if params["enable_board"] and (epoch + 1) % params["log_fig_epoch_period"] == 0:
                writer = boardfunc.log_kernels(writer, epoch, net)
                writer = boardfunc.log_data_inputrec(writer, epoch, y, yhat, params["model_distribution"])
                if params["code_supp"]:
                    writer = boardfunc.log_data_code(writer, epoch, xhat, x)
                else:
                    writer = boardfunc.log_data_code(writer, epoch, xhat, x) if torch.sum(torch.abs(x)) > 0 else boardfunc.log_data_code(writer, epoch, xhat)

            if (epoch + 1) % params["log_model_epoch_period"] == 0:
                torch.save(net, os.path.join(out_path, "model", f"model_epoch{epoch}.pt"))

        # Save final model for this file.
        torch.save(net, os.path.join(out_path, "model", "model_final.pt"))
        if params["enable_board"]:
            writer.close()


if __name__ == "__main__":
    main()
