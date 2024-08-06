import argparse
import os
import pickle
import sys

import matplotlib.pylab as plt
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname("__file__"), ".."))
sys.path.append(os.path.join(os.path.dirname("__file__"), "..", ".."))
from cindm.filepath import AIRFOILS_PATH
from cindm.model.diffusion_2d import ForceUnet, GaussianDiffusion, Trainer, Unet
from cindm.utils import reconstruct_boundary
from matplotlib.backends.backend_pdf import PdfPages
from shapely.geometry import Polygon
from torch.autograd import grad

device = torch.device("cuda:0")
normalization_filename = os.path.join(
    AIRFOILS_PATH, "training_trajectories/", "normalization_max_min.p"
)
normdict = pickle.load(open(normalization_filename, "rb"))
p_max = normdict["p_max"]
p_min = normdict["p_min"]

parser = argparse.ArgumentParser(description="inference 2d inverse design model")

parser.add_argument(
    "--batch_size", default=20, type=int, help="size of batch of input to use"
)
parser.add_argument(
    "--num_batches", default=10, type=int, help="num of batches to evaluate"
)
parser.add_argument(
    "--frames", default=6, type=int, help="number of time steps of states"
)
parser.add_argument(
    "--cond_frames",
    default=2,
    type=int,
    help="number of conditonal time steps of states",
)
parser.add_argument(
    "--pred_frames",
    default=4,
    type=int,
    help="number of prediction time steps of states",
)
parser.add_argument(
    "--ForceNet_path",
    default="../backup/checkpoint_path/force_surrogate_model.pth",
    type=str,
    help="directory of trained ForceNet checkpoint",
)
parser.add_argument(
    "--num_boundaries",
    default=2,
    type=int,
    help="number of boundaries in inverse design",
)
parser.add_argument(
    "--backward_steps",
    default=5,
    type=int,
    help="number of backward_steps for universal-backward sampling",
)
parser.add_argument(
    "--backward_lr",
    default=0.01,
    type=float,
    help="backward_lr for universal-backward sampling",
)
parser.add_argument(
    "--standard_fixed_ratio",
    default=0.01,
    type=float,
    help="standard_fixed_ratio for standard sampling",
)
parser.add_argument(
    "--forward_fixed_ratio",
    default=0.01,
    type=float,
    help="standard_fixed_ratio for universal-forward sampling",
)
parser.add_argument(
    "--coeff_ratio",
    default=0.0002,
    type=float,
    help="coeff_ratio for standard-alpha sampling, ie, lambda in the paper",
)
parser.add_argument(
    "--diffusion_model_path",
    default="./checkpoint_path/diffusion_2d/",
    type=str,
    help="directory of trained diffusion model (Unet)",
)
parser.add_argument(
    "--diffusion_checkpoint",
    default=500,
    type=int,
    help="index of checkpoint of trained diffusion model (U-Net)",
)
parser.add_argument(
    "--sum_boundary",
    default=True,
    type=bool,
    help="whether compute summed boundary when compute force",
)
parser.add_argument(
    "--share_noise",
    default=True,
    type=bool,
    help="whether share noise over different boundaries",
)
parser.add_argument(
    "--use_average_share",
    default=True,
    type=bool,
    help="whether use average when share states over different boundaries",
)
parser.add_argument(
    "--lambda_overlap",
    default=1.0,
    type=float,
    help="tradeoff between force grad and overlap grad",
)
parser.add_argument(
    "--lambda_physics",
    default=1.0,
    type=float,
    help="tradeoff between physics grad and force grad",
)
parser.add_argument(
    "--lambda_force",
    default=1.0,
    type=float,
    help="tradeoff between drag and lift force",
)
parser.add_argument(
    "--downsampling_factor",
    default=4,
    type=int,
    help="downsampling factor in computing overlap between different boundaries",
)
parser.add_argument(
    "--inference_result_path",
    default="./saved/inference_2d/",
    type=str,
    help="path to save inference result",
)

parser.add_argument(
    "--df_acc",
    default=2,
    type=int,
    help="Finite difference accuracy",
)

parser.add_argument(
    "--print_residual",
    default=False,
    type=bool,
    help="whether print all residuals",
)

parser.add_argument(
    "--save_residuals",
    default=False,
    type=bool,
    help="whether save objective residuals and physics residuals",
)

parser.add_argument(
    "--sampling_steps",
    default=1000,
    type=int,
    help="number of sampling steps",  # standard sampling steps. we change this for debugging purpose
)

args = parser.parse_args()


# unnormalize_state range from [-1, 1] to [p_min, p_max]
def unnormalize_state(pressure):
    return (0.5 * pressure + 0.5) * (p_max - p_min) + p_min


def compute_overlap(matrix):
    # matrix: [bs x num_boundary x (64*64)]
    inner_products = torch.matmul(matrix, matrix.permute(0, 2, 1))
    eye = (
        torch.eye(matrix.size(1), device=matrix.device)
        .unsqueeze(0)
        .expand(matrix.size(0), -1, -1)
    )
    inner_products *= 1 - eye
    result = inner_products.mean(dim=(-2, -1))

    return result


def force_fn(x, force_model, batch_size, num_boundaries, frames, sum_boundary=True):
    boundary = x[:, -3:]  # boundary mask and offsets
    if sum_boundary:
        boundary = (
            boundary.view(batch_size, num_boundaries, 3, 64, 64)
            .sum(dim=1, keepdim=True)
            .clamp(0.0, 1.0)
            .expand(-1, num_boundaries, -1, -1, -1)
            .reshape(batch_size * num_boundaries, 3, 64, 64)
        )
        # boundary = torch.max(boundary.view(batch_size, num_boundaries, 3, 64, 64), dim=1)[0] \
        #     .unsqueeze(1)\
        #     .clamp(0.) \
        #     .expand(-1, num_boundaries, -1, -1, -1) \
        #     .reshape(batch_size * num_boundaries, 3, 64, 64)

        pressure_boundary_pairs = [
            torch.cat(
                [unnormalize_state(x[:, 2 + 3 * i]).unsqueeze(1), boundary], dim=1
            )
            for i in range(frames)
        ]  # pressure_boundary_pairs: list of tensors of size [batch_size * num_boundaries, 4, 64, 64]
        # pressure_boundary_pairs = [torch.cat([x[:, 2 + 3 * i].unsqueeze(1), boundary], dim=1) for i in range(frames)] # pressure_boundary_pairs: list of tensors of size [batch_size * num_boundaries, 4, 64, 64]
        lift_drag_forces = [
            force_model(pressure_boundary_pairs[i]) for i in range(frames)
        ]  # lift_drag_forces: list of tensors of size [batch_size * num_boundaries, 2]
        # forces = [lift_drag_forces[i][:,1] - lift_drag_forces[i][:,0] for i in range(frames)] # force = lift (y-force) - drag (x-force). forces: list of tensors of size [batch_size * num_boundaries]. TODO, tradeoff
        # forces = [args.lambda_force * lift_drag_forces[i][:,0] + lift_drag_forces[i][:,1] for i in range(frames)] # force = lift (y-force) - drag (x-force). forces: list of tensors of size [batch_size * num_boundaries]. TODO, tradeoff
        forces = [
            args.lambda_force * torch.abs(lift_drag_forces[i][:, 0])
            + lift_drag_forces[i][:, 1]
            for i in range(frames)
        ]  # force = lift (y-force) - drag (x-force). forces: list of tensors of size [batch_size * num_boundaries]. TODO, tradeoff

        summed_forces = torch.sum(
            torch.stack(forces, dim=0), dim=0
        )  # sum over time steps
        # summed_forces = -summed_forces # minimize force #[batch_size * num_boundaries]
        grad_force = grad(
            summed_forces, x, grad_outputs=torch.ones_like(summed_forces)
        )[0]
    # if sum_boundary == False: first compute force on each boundary, then compute the sum force over different boundaries
    else:
        pressure_boundary_pairs = [
            torch.cat([x[:, 2 + 3 * i].unsqueeze(1), boundary], dim=1)
            for i in range(frames)
        ]  # pressure_boundary_pairs: list of tensors of size [batch_size * num_boundaries, 4, 64, 64]
        lift_drag_forces = [
            force_model(pressure_boundary_pairs[i]) for i in range(frames)
        ]  # lift_drag_forces: list of tensors of size [batch_size * num_boundaries, 2]
        # forces = [lift_drag_forces[i][:,1] - lift_drag_forces[i][:,0] for i in range(frames)] # force = lift (y-force) - drag (x-force). forces: list of tensors of size [batch_size * num_boundaries]. TODO, tradeoff
        forces = [
            args.lambda_force * torch.abs(lift_drag_forces[i][:, 0])
            + lift_drag_forces[i][:, 1]
            for i in range(frames)
        ]  # force = lift (y-force) - drag (x-force). forces: list of tensors of size [batch_size * num_boundaries]. TODO, tradeoff
        forces = [force.view(batch_size, num_boundaries) for force in forces]
        summed_forces = (
            torch.sum(torch.sum(torch.stack(forces, dim=0), dim=0), dim=1, keepdim=True)
            .expand(-1, num_boundaries)
            .reshape(batch_size * num_boundaries)
        )  # sum over time steps and then over boundaries
        # summed_forces = -summed_forces # minimize force #[batch_size, num_boundaries]
        grad_force = grad(
            summed_forces, x, grad_outputs=torch.ones_like(summed_forces)
        )[0]

    # Calculate drag to lift ratio
    total_drag = torch.sum(
        torch.stack([torch.abs(f[:, 0]) for f in lift_drag_forces]), dim=0
    )
    total_lift = torch.sum(torch.stack([f[:, 1] for f in lift_drag_forces]), dim=0)
    lift_to_drag_ratio = total_lift / (total_drag + 1e-8)

    return grad_force, summed_forces, lift_to_drag_ratio


def overlap_fn(x, batch_size, num_boundaries, downsampling_factor=4):
    x = x.view(batch_size, num_boundaries, -1, 64, 64)
    bd_mask = x[:, :, -3].clamp(
        0.0, 1.0
    )  # bd_mask: [batch_size, num_boundaries, 64, 64]
    new_resolution = int(64 / downsampling_factor)
    downsampled_mask = (
        bd_mask.view(
            batch_size,
            num_boundaries,
            new_resolution,
            downsampling_factor,
            new_resolution,
            downsampling_factor,
        )
        .mean(dim=(3, 5))
        .view(batch_size, num_boundaries, -1)
    )
    summed_overlap = compute_overlap(downsampled_mask)
    grad_overlap = grad(
        summed_overlap, x, grad_outputs=torch.ones_like(summed_overlap)
    )[0]

    return grad_overlap.view(batch_size * num_boundaries, -1, 64, 64)


# will be push into utils
def mask_denoise(tensor, thre=0.5):
    binary_tensor = torch.where(tensor > thre, torch.tensor(1), torch.tensor(0))
    return binary_tensor


def load_model(args):
    inp_dim = args.frames * 3 + 3
    model = Unet(
        dim=64,
        dim_mults=(1, 2),  # local Unet
        channels=inp_dim,
    )

    # load force_model
    force_model = ForceUnet(dim=64, dim_mults=(1, 2, 4, 8), channels=4)
    force_model.load_state_dict(torch.load(args.ForceNet_path))
    force_model.to(device)

    # load diffusion model
    diffusion = GaussianDiffusion(
        model,
        image_size=64,
        cond_frames=args.cond_frames,
        frames=args.frames,
        timesteps=args.sampling_steps,  # number of steps
        # sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        sampling_timesteps=args.sampling_steps,  # do not use ddim
        loss_type="l2",  # L1 or L2
        objective="pred_noise",
        diffuse_cond=True,  # diffuse on both cond states, pred states and boundary
        backward_steps=args.backward_steps,
        backward_lr=args.backward_lr,  # used in universal-backward sampling
        standard_fixed_ratio=args.standard_fixed_ratio,  # used in standard sampling
        forward_fixed_ratio=args.forward_fixed_ratio,  # used in universal forward sampling
        coeff_ratio=args.coeff_ratio,  # used in standard-alpha sampling
        share_noise=args.share_noise,
        use_average_share=args.use_average_share,
    )

    # load trainer
    trainer = Trainer(
        diffusion,
        "naca_ellipse",
        args.cond_frames,
        args.pred_frames,
        4,
        train_batch_size=48,
        train_lr=1e-4,
        train_num_steps=700000,  # total training steps
        gradient_accumulate_every=1,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        results_folder=args.diffusion_model_path,  # diffuse on both cond states, pred states and boundary
        amp=False,  # turn on mixed precision
        calculate_fid=False,  # whether to calculate fid during training
    )

    trainer.load(args.diffusion_checkpoint)  #

    # instantiate physics prior
    from cindm.model.physics_prior import DivergenceFreePrior

    physics_prior = DivergenceFreePrior(
        fd_acc=args.df_acc,
        pixels_per_dim=diffusion.image_size,
        pixels_at_boundary=False,
        reverse_d1=False,
        device=device,
    )

    # define design_fn, which takes x as input and returns the gradient of the loss w.r.t. x
    def design_fn(x):
        x.requires_grad_()
        total_grad = 0
        losses = {}
        LAMBDA_THRESHOLD = 1e-10
        # Objective function (force) loss
        grad_force, summed_force, lift_to_drag_ratio = force_fn(
            x,
            force_model,
            args.batch_size,
            args.num_boundaries,
            args.frames,
            args.sum_boundary,
        )
        losses["force"] = summed_force
        losses["lift_to_drag_ratio"] = lift_to_drag_ratio

        if np.abs(args.lambda_force) > LAMBDA_THRESHOLD:
            total_grad += args.lambda_force * grad_force
        if args.print_residual:
            print("Force residual: ", torch.mean(summed_force, dim=0))

        # Overlap loss
        grad_nonoverlap = overlap_fn(
            x, args.batch_size, args.num_boundaries, args.downsampling_factor
        )
        losses["overlap"] = grad_nonoverlap  # This is already a loss-like quantity
        if np.abs(args.lambda_overlap) > LAMBDA_THRESHOLD:
            total_grad += args.lambda_overlap * grad_nonoverlap

        # Physics loss
        grad_physics, physics_residual, physics_error = (
            physics_prior.residual_correction(x)
        )
        losses["physics"] = physics_residual
        losses["physics_error"] = physics_error

        if np.abs(args.lambda_physics) > LAMBDA_THRESHOLD:
            total_grad += args.lambda_physics * grad_physics
        if args.print_residual:
            print("Divergence residual MSE loss: ", torch.mean(physics_residual, dim=0))
            print("Divergence residual error: ", torch.mean(physics_error, dim=0))

        return total_grad, losses

    return force_model, diffusion, design_fn


def inference(force_model, diffusion, design_fn, args):
    print("start inference ...")
    design_guidance_list = ["standard-alpha"]
    lambda_str = f"force_{args.lambda_force:.2e}_overlap_{args.lambda_overlap:.2e}_physics_{args.lambda_physics:.2e}"

    result_path = os.path.join(
        args.inference_result_path,
        "num_bd_{}_frames_{}_coeff_ratio_{}_ckpt_{}_lambda_{}".format(
            args.num_boundaries,
            args.frames,
            args.coeff_ratio,
            args.diffusion_checkpoint,
            lambda_str,
        ),
    )
    # save sampling results
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    all_force_residuals = {dg: [] for dg in design_guidance_list}
    all_physics_errors = {dg: [] for dg in design_guidance_list}
    all_lift_to_drag_ratios = {dg: [] for dg in design_guidance_list}

    for batch_id in range(args.num_batches):
        print("batch_id: ", batch_id)
        preds = {}
        force_residuals = {}
        physics_errors = {}
        lift_to_drag_ratios = {}

        result_path_batch = os.path.join(result_path, "batch_{}".format(batch_id))
        if not os.path.exists(result_path_batch):
            os.makedirs(result_path_batch)
        for design_guidance in design_guidance_list:
            print(design_guidance)
            pred = diffusion.sample(
                batch_size=args.batch_size,
                design_fn=design_fn,
                design_guidance=design_guidance,
                num_boundaries=args.num_boundaries,
            )
            preds[design_guidance] = pred
            if args.save_residuals:
                x_0 = pred.reshape(-1, *pred.shape[2:])
                _, losses = design_fn(x_0)

                force_residual, physics_error, lift_to_drag_ratio = (
                    losses["force"],
                    losses["physics_error"],
                    losses["lift_to_drag_ratio"],
                )
                force_residuals[design_guidance] = force_residual.detach().cpu().numpy()
                physics_errors[design_guidance] = physics_error.detach().cpu().numpy()
                lift_to_drag_ratios[design_guidance] = (
                    lift_to_drag_ratio.detach().cpu().numpy()
                )

                all_force_residuals[design_guidance].extend(
                    force_residuals[design_guidance]
                )
                all_physics_errors[design_guidance].extend(
                    physics_errors[design_guidance]
                )
                all_lift_to_drag_ratios[design_guidance].extend(
                    lift_to_drag_ratios[design_guidance]
                )

                print(
                    f"Force Residual: {force_residual.mean().item():.4f} ± {force_residual.std().item():.4f}"
                )
                print(
                    f"Physics Error: {physics_error.mean().item():.4f} ± {physics_error.std().item():.4f}"
                )
                print(
                    f"Lift-to-Drag Ratio: {lift_to_drag_ratio.mean().item():.4f} ± {lift_to_drag_ratio.std().item():.4f}"
                )

        with open(os.path.join(result_path_batch, "preds.pkl"), "wb") as file:
            pickle.dump(preds, file)

        if args.save_residuals:
            with open(
                os.path.join(result_path_batch, "force_residuals.pkl"), "wb"
            ) as file:
                pickle.dump(force_residuals, file)
            with open(
                os.path.join(result_path_batch, "physics_errors.pkl"), "wb"
            ) as file:
                pickle.dump(physics_errors, file)
            with open(
                os.path.join(result_path_batch, "lift_to_drag_ratios.pkl"), "wb"
            ) as file:
                pickle.dump(lift_to_drag_ratios, file)

    # Save all residuals to text files
    if args.save_residuals:
        save_summary_statistics(
            result_path,
            design_guidance_list,
            all_force_residuals,
            all_physics_errors,
            all_lift_to_drag_ratios,
        )

    print("All runs completed.")


def save_summary_statistics(
    result_path,
    design_guidance_list,
    all_force_residuals,
    all_physics_residuals,
    all_lift_to_drag_ratios,
):
    summary_file = os.path.join(result_path, "summary_statistics.txt")
    with open(summary_file, "w") as f:
        for design_guidance in design_guidance_list:
            force_residuals = np.array(all_force_residuals[design_guidance])
            physics_residuals = np.array(all_physics_residuals[design_guidance])
            lift_to_drag_ratios = np.array(all_lift_to_drag_ratios[design_guidance])

            f.write(f"{design_guidance}:\n")
            for name, data in [
                ("Force Residual", force_residuals),
                ("Physics Residual", physics_residuals),
                ("Lift-to-Drag Ratio", lift_to_drag_ratios),
            ]:
                f.write(f"{name}:\n")
                f.write(f"  Mean ± Std: {np.mean(data):.4f} ± {np.std(data):.4f}\n")
                f.write(f"  Min: {np.min(data):.4f}\n")
                f.write(f"  Max: {np.max(data):.4f}\n")
                f.write(f"  Median: {np.median(data):.4f}\n")
            f.write("\n")

    print(f"Summary statistics saved to {summary_file}")


def do_overlap(boundaries):
    if len(boundaries) < 2:
        return False
    boundary_polygons = [Polygon(boundary) for boundary in boundaries]
    n = len(boundary_polygons)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if boundary_polygons[i].intersects(boundary_polygons[j]):
                return True
    return False


def select_and_plot_boundaries(args, colors=["red", "blue", "orange"]):
    design_guidance_list = [
        "standard",
        "standard-alpha",
        "universal-forward",
        "universal-backward",
        "universal-forward-recurrence-5",
    ]

    time_step = args.frames - 1  # the final time step
    n_show_examples = args.batch_size
    lambda_str = f"force_{args.lambda_force:.2e}_overlap_{args.lambda_overlap:.2e}_physics_{args.lambda_physics:.2e}"

    result_path = os.path.join(
        args.inference_result_path,
        "num_bd_{}_frames_{}_coeff_ratio_{}_ckpt_{}_lambda_{}".format(
            args.num_boundaries,
            args.frames,
            args.coeff_ratio,
            args.diffusion_checkpoint,
            lambda_str,
        ),
    )
    assert os.path.exists(result_path)

    for batch_id in range(args.num_batches):
        print("batch_id: ", batch_id)
        result_path_batch = os.path.join(result_path, "batch_{}".format(batch_id))
        preds = pickle.load(open(os.path.join(result_path_batch, "preds.pkl"), "rb"))

        for design_guidance in design_guidance_list:
            if design_guidance not in preds:
                continue

            output = preds[design_guidance].detach()

            fname = f"./plot_state_boundaries_batch_{batch_id}_{design_guidance}.pdf"
            pdf = PdfPages(os.path.join(result_path, fname))

            valid = 0
            for i in range(args.batch_size):
                boundaries = []
                isvalid = True
                for j in range(args.num_boundaries):
                    mask = output[i][j][-3].cpu()[1:-1, 1:-1]
                    offset = output[i][j][-2:].cpu()[:, 1:-1, 1:-1].permute(1, 2, 0)
                    clean_mask = mask_denoise(mask)
                    try:
                        restored_boundary = reconstruct_boundary(clean_mask, offset)
                    except:
                        isvalid = False
                        break
                    if restored_boundary.shape[0] < 3:
                        isvalid = False
                        break
                    boundaries.append(restored_boundary)

                if not isvalid or do_overlap(boundaries):
                    continue

                fig, ax = plt.subplots(
                    figsize=(args.num_boundaries * 12 + 5, 4),
                    ncols=2 * args.num_boundaries + 1,
                )
                if not os.path.exists(os.path.join(result_path, "boundaries")):
                    os.makedirs(os.path.join(result_path, "boundaries"))

                for j in range(args.num_boundaries):
                    bd = [[x[0], x[1], 0.0] for x in boundaries[j].tolist()]
                    with open(
                        os.path.join(
                            result_path,
                            "boundaries",
                            f"batch_{batch_id}_sim_{valid}_boundary_{j}_{design_guidance}.txt",
                        ),
                        "w",
                    ) as file:
                        file.write(str(bd))

                    vx = output[i][j][time_step * 3].cpu().numpy()
                    im = ax[j * 2].imshow(vx, cmap="viridis", aspect="auto")
                    ax[j * 2].set_title(f"{design_guidance}: Example {i}, bd {j} v_x")
                    ax[j * 2].set_xlabel("X-Axis")
                    ax[j * 2].set_ylabel("Y-Axis")
                    ax[j * 2].set_ylim(0, 62)
                    ax[j * 2].set_xlim(0, 62)

                    vy = output[i][j][1 + time_step * 3].cpu().numpy()
                    im = ax[j * 2 + 1].imshow(vy, cmap="viridis", aspect="auto")
                    ax[j * 2 + 1].set_title(
                        f"{design_guidance}: Example {i}, bd {j} v_y"
                    )
                    ax[j * 2 + 1].set_xlabel("X-Axis")
                    ax[j * 2 + 1].set_ylabel("Y-Axis")
                    ax[j * 2 + 1].set_ylim(0, 62)
                    ax[j * 2 + 1].set_xlim(0, 62)

                    for k in range(2):
                        ax[j * 2 + k].plot(
                            np.append(boundaries[j][:, 0], boundaries[j][0, 0]),
                            np.append(boundaries[j][:, 1], boundaries[j][0, 1]),
                            color=colors[j],
                        )
                        fig.colorbar(im, ax=ax[j * 2 + k])

                    ax[2 * args.num_boundaries].plot(
                        np.append(boundaries[j][:, 0], boundaries[j][0, 0]),
                        np.append(boundaries[j][:, 1], boundaries[j][0, 1]),
                        color=colors[j],
                    )

                ax[2 * args.num_boundaries].set_xlim(0, 62)
                ax[2 * args.num_boundaries].set_ylim(0, 62)
                ax[2 * args.num_boundaries].set_title(
                    f"{design_guidance}: Example {i}, {args.num_boundaries} boundaries together"
                )
                valid += 1
                pdf.savefig(fig)
                plt.close(fig)

            pdf.close()
            print(f"Completed {design_guidance} for batch {batch_id}")


def select_and_plot_boundaries_new(
    args,
    colors=["red", "blue", "orange"],
    var_to_plot=["vx", "vy", "p"],  # only velocities are plotted
):
    design_guidance_list = [
        "standard",
        "standard-alpha",
        "universal-forward",
        "universal-backward",
        "universal-forward-recurrence-5",
    ]

    time_step = args.frames - 1  # the final time step
    n_show_examples = args.batch_size
    num_var_to_plot = len(var_to_plot)

    lambda_str = f"force_{args.lambda_force:.2e}_overlap_{args.lambda_overlap:.2e}_physics_{args.lambda_physics:.2e}"

    result_path = os.path.join(
        args.inference_result_path,
        "num_bd_{}_frames_{}_coeff_ratio_{}_ckpt_{}_lambda_{}".format(
            args.num_boundaries,
            args.frames,
            args.coeff_ratio,
            args.diffusion_checkpoint,
            lambda_str,
        ),
    )
    assert os.path.exists(result_path)

    for batch_id in range(args.num_batches):
        print("batch_id: ", batch_id)
        result_path_batch = os.path.join(result_path, "batch_{}".format(batch_id))
        preds = pickle.load(open(os.path.join(result_path_batch, "preds.pkl"), "rb"))

        for design_guidance in design_guidance_list:
            if design_guidance not in preds:
                continue

            output = preds[design_guidance].detach()

            fname = f"./plot_state_boundaries_batch_{batch_id}_{design_guidance}.pdf"
            pdf = PdfPages(os.path.join(result_path, fname))

            valid = 0
            for i in range(args.batch_size):
                boundaries = []
                isvalid = True
                for j in range(args.num_boundaries):
                    mask = output[i][j][-3].cpu()[1:-1, 1:-1]
                    offset = output[i][j][-2:].cpu()[:, 1:-1, 1:-1].permute(1, 2, 0)
                    clean_mask = mask_denoise(mask)
                    try:
                        restored_boundary = reconstruct_boundary(clean_mask, offset)
                    except:
                        isvalid = False
                        break
                    if restored_boundary.shape[0] < 3:
                        isvalid = False
                        break
                    boundaries.append(restored_boundary)

                if not isvalid or do_overlap(boundaries):
                    continue

                fig, ax = plt.subplots(
                    figsize=(num_var_to_plot * 5 + 5, 4), ncols=num_var_to_plot + 1
                )

                for j, var in enumerate(var_to_plot):
                    if var == "vx":
                        field = output[i][0][time_step * 3].cpu().numpy()
                    elif var == "vy":
                        field = output[i][0][1 + time_step * 3].cpu().numpy()
                    elif var == "p":
                        field = output[i][0][2 + time_step * 3].cpu().numpy()
                    else:
                        raise ValueError(f"Unsupported variable: {var}")

                    im = ax[j].imshow(field, cmap="viridis", aspect="auto")
                    ax[j].set_title(f"{design_guidance}: Example {i}, {var}")
                    ax[j].set_xlabel("X-Axis")
                    ax[j].set_ylabel("Y-Axis")
                    ax[j].set_ylim(0, 62)
                    ax[j].set_xlim(0, 62)
                    fig.colorbar(im, ax=ax[j])

                    # Overlay boundaries on each field plot
                    for k in range(args.num_boundaries):
                        ax[j].plot(
                            np.append(boundaries[k][:, 0], boundaries[k][0, 0]),
                            np.append(boundaries[k][:, 1], boundaries[k][0, 1]),
                            color=colors[k],
                        )

                # Plot showing only boundaries
                ax[-1].set_xlim(0, 62)
                ax[-1].set_ylim(0, 62)
                ax[-1].set_title(f"{design_guidance}: Example {i}, All Boundaries")
                ax[-1].set_xlabel("X-Axis")
                ax[-1].set_ylabel("Y-Axis")
                for k in range(args.num_boundaries):
                    ax[-1].plot(
                        np.append(boundaries[k][:, 0], boundaries[k][0, 0]),
                        np.append(boundaries[k][:, 1], boundaries[k][0, 1]),
                        color=colors[k],
                    )

                # Save boundaries to text files
                if not os.path.exists(os.path.join(result_path, "boundaries")):
                    os.makedirs(os.path.join(result_path, "boundaries"))

                for k in range(args.num_boundaries):
                    bd = [[x[0], x[1], 0.0] for x in boundaries[k].tolist()]
                    with open(
                        os.path.join(
                            result_path,
                            "boundaries",
                            f"batch_{batch_id}_sim_{i}_boundary_{k}_{design_guidance}.txt",
                        ),
                        "w",
                    ) as file:
                        file.write(str(bd))

                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        result_path,
                        f"batch_{batch_id}_example_{i}_{design_guidance}.png",
                    )
                )
                pdf.savefig(fig)
                plt.close(fig)

            pdf.close()
            print(f"Completed {design_guidance} for batch {batch_id}")


def main(args):
    force_model, diffusion, design_fn = load_model(args)
    inference(force_model, diffusion, design_fn, args)
    # select_and_plot_boundaries(args)
    select_and_plot_boundaries_new(args)


# run all the inference experiments
def meta_main(tune_args_list, args):
    for i, tune_args in enumerate(tune_args_list):
        args.num_boundaries = tune_args["num_boundaries"]
        args.coeff_ratio = tune_args["coeff_ratio"]

        print(i, "new args: ", args)
        main(args)


if __name__ == "__main__":
    # get_ipython().run_line_magic('matplotlib', 'inline')
    # args = parser.parse_args([])

    # args.num_batches = 10
    # args.batch_size = 20
    # args.cond_frames = 2
    # args.pred_frames = 4
    # args.frames = args.cond_frames + args.pred_frames
    # args.coeff_ratio = 0.01
    # args.standard_fixed_ratio = 0.001
    # args.forward_fixed_ratio = 0.1
    # args.backward_steps = 5
    # args.backward_lr = 0.01
    # args.num_boundaries = 3
    # args.downsampling_factor=4
    # args.sum_boundary = True
    # args.share_noise = True
    # args.use_average_share = True
    # args.lambda_force = 1.
    # args.lambda_overlap = 1.
    # # args.diffusion_model_path = "./checkpoint_path/diffusion_2d/"
    # args.diffusion_checkpoint = 500
    # args.inference_result_path = "./saved/inference_2d/"

    tune_args_list = [{"num_boundaries": args.num_boundaries, "coeff_ratio": 0.0002}]
    meta_main(tune_args_list, args)
    # main(args)
