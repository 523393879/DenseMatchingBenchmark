import torch
import torch.nn.functional as F

from dmb.modeling.stereo.layers.inverse_warp_3d import inverse_warp_3d


def correlation_1d(reference_fm, target_fm, max_disp=192, start_disp=0, dilation=1, disp_sample=None):
    device = reference_fm.device
    B, C, H, W = reference_fm.shape

    if disp_sample is None:
        end_disp = start_disp + max_disp - 1

        disp_sample_number = (max_disp + dilation - 1) // dilation
        D = disp_sample_number

        # generate disparity samples, in [B,D, H, W] layout
        disp_sample = torch.linspace(start_disp, end_disp, D)
        disp_sample = disp_sample.view(1, D, 1, 1).expand(B, D, H, W).to(device).float()

    else: # direct provide disparity samples
        # the number of disparity samples
        D = disp_sample.shape[1]

    # expand D dimension
    concat_reference_fm = reference_fm.unsqueeze(2).expand(B, C, D, H, W)
    concat_target_fm = target_fm.unsqueeze(2).expand(B, C, D, H, W)

    # shift target feature according to disparity samples
    # [B, C, D, H, W]
    concat_target_fm = inverse_warp_3d(concat_target_fm, -disp_sample, padding_mode='zeros')

    # mask out features in reference
    # [B, C, D, H, W]
    concat_reference_fm = concat_reference_fm * (concat_target_fm > 0).float()

    # [B, D, H, W]
    correlation_1d_cost = (concat_reference_fm * concat_reference_fm).sum(dim=1, keepdim=False)

    # [B, D, H, W]
    correlation_1d_cost = F.leaky_relu(correlation_1d_cost, negative_slope=0.1, inplace=True)

    return correlation_1d_cost

COR_FUNCS = dict(
    default=correlation_1d,
)
