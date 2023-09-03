import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from core.utils.skeleton_utils import (
    rasterize_points
)

from typing import Mapping, Any, Optional, Callable


class BaseLoss(nn.Module):
    def __init__(self, weight: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class NeRFRGBMSELoss(BaseLoss):

    def __init__(self, fine: float = 1.0, coarse: float = 1.0, **kwargs):
        super(NeRFRGBMSELoss, self).__init__(**kwargs)
        self.fine = fine
        self.coarse = coarse

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):

        rgb_pred = preds['rgb_map']
        loss_fine = (rgb_pred - batch['target_s']).pow(2.).mean()

        loss_coarse = torch.tensor(0.0)

        if 'rgb0' in preds:
            rgb_pred = preds['rgb0']
            loss_coarse = (rgb_pred - batch['target_s']).pow(2.).mean()
        
        loss = loss_fine * self.fine + loss_coarse * self.coarse

        return loss, {'loss_fine': loss_fine.item(), 'loss_coarse': loss_coarse.item()}


class NeRFRGBLoss(BaseLoss):

    def __init__(self, fine: float = 1.0, coarse: float = 1.0, **kwargs):
        super(NeRFRGBLoss, self).__init__(**kwargs)
        self.fine = fine
        self.coarse = coarse

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):

        rgb_pred = preds['rgb_map']
        loss_fine = (rgb_pred - batch['target_s']).abs().mean()

        loss_coarse = torch.tensor(0.0)

        if 'rgb0' in preds:
            rgb_pred = preds['rgb0']
            loss_coarse = (rgb_pred - batch['target_s']).abs().mean()
        
        loss = loss_fine * self.fine + loss_coarse * self.coarse

        return loss, {'loss_fine': loss_fine.item(), 'loss_coarse': loss_coarse.item()}



class SoftSoftmaxLoss(BaseLoss):

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], model: Callable, **kwargs):

        a = preds['agg_logit']
        labels = ((preds['T_i'] * preds['alpha']) > 0).float()

        vol_invalid = preds['vol_invalid']
        vol_valid = 1 - vol_invalid

        p = model.get_agg_weights(a, vol_invalid, mask_invalid=False)
        p_valid = (p * vol_valid).sum(-1)
        soft_softmax_loss = (labels - p_valid).pow(2.).mean()

        loss = soft_softmax_loss * self.weight 

        valid_count = ((vol_valid.sum(-1) > 0) * labels).sum()
        sigmoid_act = (p * vol_valid).sum(-1) * labels
        act_avg = sigmoid_act.detach().sum() / valid_count
        act_max = sigmoid_act.detach().max()

        loss_stats = {
            'soft_softmax_loss': soft_softmax_loss.item(), 
            'sigmoid_avg_act': act_avg.item(), 
            'sigmoid_max_act': act_max.item(),
        } 
        return loss, loss_stats


class VolScaleLoss(BaseLoss):

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        vol_scale = preds['vol_scale']
        valid = ((1 - preds['vol_invalid']).reshape(-1, len(vol_scale))).sum(dim=0) > 0
        scale_loss = (torch.prod(vol_scale, dim=-1) * valid).sum() 
        loss = scale_loss * self.weight

        scale_avg = vol_scale.detach().mean(0)
        scale_x, scale_y, scale_z = scale_avg
        
        return loss, {'opt_scale_x': scale_x, 'opt_scale_y': scale_y, 'opt_scale_z': scale_z}
    

class EikonalLoss(BaseLoss):

    def __init__(self, *args, use_valid: bool = True, **kwargs):
        self.use_valid = use_valid
        super(EikonalLoss, self).__init__(*args, **kwargs)

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        if 'vol_invalid' in preds:
            valid = (1 - preds['vol_invalid']).sum(-1).clamp(max=1.0)
        else:
            valid = torch.ones(preds['surface_gradient'].shape[:-1])
        
        
        if self.use_valid:
            norm = preds['surface_gradient'][valid > 0].norm(dim=-1)
            eikonal_loss = (norm - 1).pow(2.).sum() / valid.sum()
        else:
            norm = preds['surface_gradient'].norm(dim=-1)
            eikonal_loss = (norm - 1).pow(2.).mean()

        loss = eikonal_loss * self.weight

        return loss, {'eikonal_loss': eikonal_loss, 'surface_normal': norm.mean().item()}


class PointDeformLoss(BaseLoss):

    def __init__(self, *args, weight: float = 0.1, threshold: float = 0.04, **kwargs):
        '''
        threshold in meter
        '''
        self.threshold = threshold
        super(PointDeformLoss, self).__init__(*args, weight=weight, **kwargs)

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        vol_scale = preds['vol_scale'].reshape(1, -1, 1, 3)
        # in m-unit space
        if batch['device_cnt'] > 1:
            vol_scale = torch.chunk(vol_scale, chunks=batch['device_cnt'], dim=1)[0]
        dp = preds['dp'] if not ('dp_uc' in preds) else preds['dp_uc']

        N_joints = dp.shape[1]

        # TODO: hard coded
        threshold = torch.ones(1, N_joints, 1) * self.threshold
        mag = dp.pow(2.).sum(dim=-1)
        mag = torch.where((mag + 1e-6).pow(0.5) > threshold, mag, torch.zeros_like(mag))
        point_deform_loss = mag.mean()
        loss = point_deform_loss * self.weight

        return loss, {'point_deform_loss': point_deform_loss, 
                      'max_deform_norm': dp.norm(dim=-1).max(), 
                      'mean_deform_norm': dp.norm(dim=-1).mean(), 
                      'median_deform_norm': dp.norm(dim=-1).median()
                     }


class PointCloudsEikonalLoss(BaseLoss):

    def __init__(self, *args, weight: float = 0.001, **kwargs):
        '''
        when schedule=True, linearly scale the weight to weight
        '''
        super().__init__()
        self.weight_ = weight
    

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        weight = self.weight_
        if weight == 0.:
            return torch.tensor(0.), {}
        pc_grad = preds['pc_grad']

        pc_norm = pc_grad.norm(dim=-1)
        pc_eikonal_loss = (pc_norm - 1).pow(2.).mean()
        loss = pc_eikonal_loss * weight

        return loss, {'pc_eikonal_loss': pc_eikonal_loss}


class PointCloudsSurfaceLoss(BaseLoss):

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        if self.weight == 0.:
            return torch.tensor(0.), {}
        sdf = preds['pc_sigma']
        mask = (sdf > 0).float()
        pc_surface_loss = (sdf * mask).mean()
        # anchor points are on the surface -> sigma = 0
        loss = pc_surface_loss * self.weight

        return loss, {'pc_surface_loss': pc_surface_loss}


class PointCloudsNeighborLoss(BaseLoss):
    """
    distance between neighboring points should be preserved
    """
    def __init__(self, *args, weight: float = 100., **kwargs):
        super().__init__()
        self.weight_ = weight

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        weight = self.weight_
        if weight == 0.:
            return torch.tensor(0.), {}
        # deformed anchor location in world space
        p_w = preds['p_w'] if not 'p_w_uc' in preds else preds['p_w_uc']

        N_graphs, N_joints, N_pts, _ = p_w.shape

        if batch['device_cnt'] > 1:
            preds['nb_idxs'] = preds['nb_idxs'][:N_joints]
            preds['nb_diffs'] = preds['nb_diffs'][:N_joints]

        # neighbors for each point
        # note: the first one is always the point itself
        nb_idxs = preds['nb_idxs'][..., 1:]
        nb_dists = preds['nb_diffs'][..., 1:, :].norm(dim=-1)

        nb_pts = p_w.reshape(N_graphs, -1, 3)[:, nb_idxs]

        deform_dists = (nb_pts - p_w[..., None, :]).norm(dim=-1)
        dist_loss = (deform_dists - nb_dists).pow(2.).mean()
        loss = dist_loss * weight
        return loss, {'pc_dist_loss': dist_loss}


class BkgdLoss(BaseLoss):

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        if self.weight == 0.:
            return torch.tensor(0.), {}
        bkgd_map = (1 - preds['acc_map'])[..., None].detach()
        pred_bgs = preds['bg_preds']
        target = batch['target_s_not_masked']

        bkgd_loss = (bkgd_map * (pred_bgs - target).pow(2.)).mean()
        loss = bkgd_loss * self.weight
        return loss, {'bkgd_loss': bkgd_loss}


class SigmaLoss(BaseLoss):

    def forward(self, batch: Mapping[str, Any], preds: Mapping[str, Any], **kwargs):
        T_i = preds['T_i']
        logp = torch.exp(-T_i.abs()) + torch.exp(-(1-T_i).abs())
        sigma_loss = -logp.mean() 
        loss = sigma_loss * self.weight

        return loss, {'sigma_loss': sigma_loss}

