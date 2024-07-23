import torch
from grad_utils import *


class DivergenceFreePrior:
    def __init__(
        self,
        fd_acc,
        pixels_per_dim,
        pixels_at_boundary,
        reverse_d1,
        guidance_strength=1.0,
        device="cpu",
        bcs="none",
        normalizer=None,
        domain_length=1.0,
        residual_grad_guidance=False,
        use_ddim_x0=False,
        ddim_steps=0,
        residual_type="mse",  # MSE
        residual_scaled=False,  # residual scaled
    ):
        """
        Initialize the residual evaluation for divergence-free constraints

        :param fd_acc: Finite difference accuracy.
        :param pixels_per_dim: Number of pixels per dimension.
        :param pixels_at_boundary: Whether to have pixels at the boundary.
        :param reverse_d1: Whether to reverse the second dimension.
        """

        self.pixels_at_boundary = pixels_at_boundary
        self.periodic = False
        self.input_dim = 2
        self.guidance_strength = guidance_strength
        self.residual_type = residual_type
        self.residual_scaled = residual_scaled
        self.normalizer = normalizer if normalizer is not None else None

        if bcs == "periodic":
            self.periodic = True

        if self.pixels_at_boundary:
            d0 = domain_length / (pixels_per_dim - 1)
            d1 = domain_length / (pixels_per_dim - 1)
        else:
            d0 = domain_length / pixels_per_dim
            d1 = domain_length / pixels_per_dim

        self.reverse_d1 = reverse_d1
        if self.reverse_d1:
            d1 *= -1.0  # this is for later consistency with visualization

        self.grads = GradientsHelper(
            d0=d0, d1=d1, fd_acc=fd_acc, periodic=self.periodic, device=device
        )

        # we only need to compute the divergence of the velocity field

    def forward(
        self,
        input,
    ):
        assert (
            len(input.shape) == 4
        ), "Model output must be a tensor shaped as an image (with explicit axes for the spatial dimensions)."
        batch_size, output_dim, pixels_per_dim, pixels_per_dim = (
            input.shape
        )  # (B, T, H, W), where T = (3*t + 3)

        if self.normalizer:
            input = self.normalizer.decode(input)  # unnormalize the input
        else:
            pass

        u_fields = input[:, 0::3, :, :]  # x velocity (B, t, H, W)
        v_fields = input[:, 1::3, :, :]  # y velocity (B, t, H, W)

        u_d0 = self.grads.stencil_gradients(u_fields, mode="d_d0")
        v_d1 = self.grads.stencil_gradients(v_fields, mode="d_d1")

        # compute divergence residual
        div_res = u_d0 + v_d1  # (B, T, H, W)
        difference = div_res - torch.zeros_like(div_res)  # Divergence-free constraint

        if self.residual_type == "l2":
            # L2 norm: root mean square of the differences
            residual = torch.sqrt(torch.mean(difference**2))
        elif self.residual_type == "l1":
            # L1 norm: mean absolute value of the differences
            residual = torch.mean(torch.abs(difference))
        elif self.residual_type == "mse":
            # Mean squared error
            residual = torch.mean(difference**2)
        else:
            raise ValueError("Unsupported residual type")

        return residual

    def residual_correction(self, x0_pred):
        assert (
            len(x0_pred.shape) == 4
        ), "Model output must be a tensor shaped as b_c_xy."

        residual = self.forward(x0_pred)
        grad = torch.autograd.grad(residual, x0_pred)[0]

        max_dr_dx = torch.tensor(torch.max(grad)).to(x0_pred.device)
        max_dr_dx = torch.clamp(max_dr_dx, max=1e12)
        correction_eps = self.guidance_strength / max_dr_dx
        residual_grad = correction_eps * grad.detach()
        # residual_corrected = self.forward(x0_pred)

        return residual_grad, residual
