import torch
from grad_utils import *
import einops as ein
        
class DivergenceFreePrior:
    def __init__(self, model, fd_acc, pixels_per_dim, pixels_at_boundary, reverse_d1, 
                 guidance_strength,
                 device = 'cpu',
                 bcs = 'none',
                 domain_length = 1.,
                 residual_grad_guidance = False,
                 use_ddim_x0 = False, ddim_steps = 0):
        """
        Initialize the residual evaluation for divergence-free constraints

        :param model: The neural network model to compute the residuals for.
        :param fd_acc: Finite difference accuracy.
        :param pixels_per_dim: Number of pixels per dimension.
        :param pixels_at_boundary: Whether to have pixels at the boundary.
        :param reverse_d1: Whether to reverse the second dimension.
        :param n_steps: Number of steps for time discretization.
        :param E: Young's Modulus.
        :param nu: Poisson's Ratio.
        """
        self.gov_eqs = 'darcy'
        self.model = model
        self.pixels_at_boundary = pixels_at_boundary
        self.periodic = False
        self.input_dim = 2
        self.guidance_strength = guidance_strength
        
        if bcs == 'periodic':
            self.periodic = True

        if self.pixels_at_boundary:
            d0 = domain_length / (pixels_per_dim - 1)
            d1 = domain_length / (pixels_per_dim - 1)
        else:
            d0 = domain_length / pixels_per_dim
            d1 = domain_length / pixels_per_dim
        
        self.reverse_d1 = reverse_d1
        if self.reverse_d1:
            d1 *= -1. # this is for later consistency with visualization

        self.grads = GradientsHelper(d0=d0, d1=d1, fd_acc = fd_acc, periodic=self.periodic, device=device)
        self.relu = torch.nn.ReLU()
        self.pixels_per_dim = pixels_per_dim
        
        # we only need to compute the divergence of the velocity field

    def compute_residual(self, input, reduce = 'none', return_model_out = False,
                         return_optimizer = False, return_inequality = False, 
                         sample = False, ddim_func = None, pass_through = False):
        
        if pass_through:
            assert isinstance(input, torch.Tensor), 'Input is assumed to directly be given output.'
            x0_pred = input
            model_out = x0_pred
        else:
            assert len(input[0]) == 2 and isinstance(input[0], tuple), 'Input[0] must be a tuple consisting of noisy signal and time.'
            noisy_in, time = iter(input[0])

            if self.residual_grad_guidance:
                assert not self.use_ddim_x0, 'Residual gradient guidance is not implemented with sample estimation for residual.'
                noisy_in.requires_grad = True
                residual_noisy_in = self.compute_residual(generalized_b_xy_c_to_image(noisy_in), pass_through = True)['residual']
                dr_dx = torch.autograd.grad(residual_noisy_in.abs().mean(), noisy_in)[0]
                if sample:
                    x0_pred = self.model.forward_with_guidance_scale(noisy_in, time, cond = dr_dx, guidance_scale = 3.) # There is no mentioning of value for the guidance scale in the paper and repo?!?
                    model_out = x0_pred
                else:
                    x0_pred = self.model(noisy_in, time, cond = dr_dx, null_cond_prob = 0.1)
                    model_out = x0_pred
            else:
                pass
        
        assert len(x0_pred.shape) == 5, 'Model output must be a tensor shaped as an image (with explicit axes for the spatial dimensions).'
        batch_size, temp_dim, output_dim, pixels_per_dim, pixels_per_dim = x0_pred.shape # (B, T, C, H, W)
        
        velocity = x0_pred[:, :, 0, :, :] # velocity fields
        v_d0 = self.grads.stencil_gradients(velocity[:, :, 0, :, :], mode='d_d0')
        u_d1 = self.grads.stencil_gradients(velocity[:, :, 1, :, :], mode='d_d1')
        
        # compute divergence residual
        div_res = v_d0 + u_d1 # (B, T, H, W)
        return div_res
        
    def residual_correction(self, x0_pred_in):
        
        # Ensure the model output is in the correct shape
        assert len(x0_pred_in.shape) == 4, 'Model output must be a tensor shaped as b_xy_c.'

        x0_pred = x0_pred_in.detach().clone()
        x0_pred.requires_grad_(True)

        residual_x0_pred = self.compute_residual(generalized_b_xy_c_to_image(x0_pred), pass_through = True)['residual']
        dr_dx = torch.autograd.grad(torch.sum(residual_x0_pred**2), x0_pred)[0][:,:,0] # residuals w.r.t. p

        max_dr_dx = torch.tensor(dr_dx).to(x0_pred.device)
        max_dr_dx = torch.clamp(max_dr_dx, max=1e12)
        correction_eps = self.guidance_strength / max_dr_dx
        x0_pred_in[:,:,0] -= correction_eps.unsqueeze(1) * dr_dx.detach()

        # compute residual again based on correction
        residual_corrected = self.compute_residual(generalized_b_xy_c_to_image(x0_pred_in), pass_through = True)['residual']
        return x0_pred_in, residual_corrected
