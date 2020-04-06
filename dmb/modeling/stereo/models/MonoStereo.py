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
        ref_fms, tgt_fms = self.backbone(ref_img, tgt_img)

        # all returned disparity map are in 1/4 resolution
        disparity_sample, proposal_disps, proposal_costs, offsets, masks = self.disp_sampler(left=ref_fms, right=tgt_fms)
        # up-sample to full resolution
        h, w = ref_img.shape[-2:]
        ph, pw = disparity_sample.shape[-2:]
        full_disparity_sample = F.interpolate(disparity_sample * w / pw, size=(h, w), mode='bilinear', align_corners=False)
        full_proposal_disps = [F.interpolate(d * w / d.shape[-1], size=(h, w), mode='bilinear', align_corners=False) for d in proposal_disps]
        full_offsets = [F.interpolate(f*w/f.shape[-1], size=(h, w), mode='bilinear', align_corners=False) for f in offsets]
        full_masks = [F.interpolate(m, size=(h, w), mode='bilinear', align_corners=False) for m in masks]


        # compute cost volume
        costs = self.cost_processor(ref_fms, tgt_fms, disp_sample=disparity_sample)

        # disparity prediction
        disps = [self.disp_predictor(cost, disp_sample=full_disparity_sample) for cost in costs]

        # disparity refinement
        if self.disp_refinement is not None:
            disps = self.disp_refinement(disps, ref_fms, tgt_fms, ref_img, tgt_img)

        # # extend disparity map estimated in monocular way
        disparity_samples = torch.split(full_disparity_sample, 1, dim=1)
        disps.extend(disparity_samples)
        # disps.extend(full_proposal_disps)

        # supervise cost computed in disparity sampler network
        costs = proposal_costs

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
            )

            gsm_loss_dict = self.loss_evaluator(disps, costs, target, **loss_args)
            loss_dict.update(gsm_loss_dict)

            return {}, loss_dict

        else:

            # visualize residual disparity map
            res_disps = []
            for i in range(1, 3):
                res_disps.append((disps[i-1] - disps[i]).abs())
            disps.extend(res_disps)

            # visualize disparity sample
            # disparity_samples = torch.split(full_disparity_sample, 1, dim=1)
            # disps.extend(disparity_samples)

            results = dict(
                disps=disps,
                costs=costs,
                offsets=full_offsets,
                masks=full_masks,
            )

            if self.cmn is not None:
                # confidence measurement network
                variance, confs = self.cmn(costs, target)
                results.update(confs=confs)

            return results, {}
