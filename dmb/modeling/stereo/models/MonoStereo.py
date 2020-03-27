import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.backbones import build_backbone
from dmb.modeling.stereo.disp_samplers import build_disp_sampler
from dmb.modeling.stereo.cost_processors import build_cost_processor
from dmb.modeling.stereo.cmn import build_cmn
from dmb.modeling.stereo.disp_predictors import build_disp_predictor
from dmb.modeling.stereo.disp_refinement import build_disp_refinement
from dmb.modeling.stereo.losses import make_gsm_loss_evaluator


class MonoStereo(nn.Module):
    """
    A general stereo matching model which fits most methods.

    """
    def __init__(self, cfg):
        super(MonoStereo, self).__init__()
        self.cfg = cfg.copy()
        self.max_disp = cfg.model.max_disp

        self.backbone = build_backbone(cfg)

        self.disp_sampler = build_disp_sampler(cfg)

        self.cost_processor = build_cost_processor(cfg)

        # confidence measurement network
        self.cmn = None
        if 'cmn' in cfg.model:
            self.cmn = build_cmn(cfg)

        self.disp_predictor = build_disp_predictor(cfg)

        self.disp_refinement = None
        if 'disp_refinement' in cfg.model:
            self.disp_refinement = build_disp_refinement(cfg)

        # make general stereo matching loss evaluator
        self.loss_evaluator = make_gsm_loss_evaluator(cfg)

    def forward(self, batch):
        # parse batch
        ref_img, tgt_img = batch['leftImage'], batch['rightImage']
        target = batch['leftDisp'] if 'leftDisp' in batch else None

        # extract image feature
        ref_group_fms, tgt_group_fms = self.backbone(ref_img, tgt_img)

        # [B, 32, H//16, W//16], [B, 32, H//8, W//8], [B, 32, H//4, W//4]
        ref_fms_16, ref_fms_8, ref_fms = ref_group_fms
        tgt_fms_16, tgt_fms_8, tgt_fms = tgt_group_fms
        # ref_fms, tgt_fms = self.backbone(ref_img, tgt_img)

        # all returned disparity map are in full resolution
        disparity_sample, mono_disps, corr_costs = self.disp_sampler(left=ref_group_fms, right=tgt_group_fms)

        # compute cost volume
        h, w = ref_fms.shape[-2:]
        down_disparity_sample = F.interpolate(disparity_sample, size=(h, w), mode='bilinear', align_corners=False)
        costs = self.cost_processor(ref_fms, tgt_fms, disp_sample=down_disparity_sample)

        # disparity prediction
        disps = [self.disp_predictor(cost, disp_sample=disparity_sample) for cost in costs]

        # disparity refinement
        if self.disp_refinement is not None:
            disps = self.disp_refinement(disps, ref_fms, tgt_fms, ref_img, tgt_img)

        # extend disparity map estimated in monocular way
        disps.extend(mono_disps)

        costs = corr_costs

        if self.training:
            loss_dict = dict()
            variance = None
            if hasattr(self.cfg.model.losses, 'focal_loss'):
                variance = self.cfg.model.losses.focal_loss.get('variance', None)

            if self.cmn is not None:
                # confidence measurement network
                variance, cm_losses = self.cmn(costs, target)
                loss_dict.update(cm_losses)

            loss_args = dict(
                variance = variance,
                # disp_sample=disparity_sample,
            )

            gsm_loss_dict = self.loss_evaluator(disps, costs, target, **loss_args)
            loss_dict.update(gsm_loss_dict)

            return {}, loss_dict

        else:

            # visualize residual disparity map
            res_disps = []
            for i in range(1, len(disps)):
                res_disps.append(disps[i-1] - disps[i])
            disps.extend(res_disps)

            results = dict(
                disps=disps,
                costs=costs,
            )

            if self.cmn is not None:
                # confidence measurement network
                variance, confs = self.cmn(costs, target)
                results.update(confs=confs)

            return results, {}
