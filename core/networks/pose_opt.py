import torch
import torch.nn as nn
import numpy as np
from core.utils.skeleton_utils import (
    Skeleton, 
    SMPLSkeleton, 
    calculate_kinematic,
)
from typing import Optional, Dict, Any
from einops import rearrange

class PoseOpt(nn.Module):
    """
    Pose optimization module

    """

    def __init__(
        self,
        rest_pose: torch.Tensor,
        kp3d: np.ndarray,
        bones: np.ndarray, # FIXME: shouldn't call it bones anymore
        skel_type: Skeleton,
        emb_dim: int = 16, # unique-identifier for the per-pose code
        depth: int = 4,
        width: int = 128,
        residual_scale: float = 0.1, # scale the output
        rot_6d: bool = False,
        n_embs: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        assert skel_type == SMPLSkeleton, 'Only SMPLSkeleton is supported for now'
        assert depth >= 2, 'Depth must be at least 2'
        self.skel_type = skel_type
        self.n_embs = n_embs if n_embs is not None else len(kp3d)
        self.residual_scale = residual_scale

        rvecs = torch.tensor(bones)
        # NOTE: this is different from original A-NeRF implementation, but should 
        # work better
        pelvis = torch.tensor(kp3d[:, skel_type.root_id])

        self.register_buffer('rest_pose', rest_pose, persistent=False)
        self.register_buffer('pelvis', pelvis, persistent=False)
        self.register_buffer('rvecs', rvecs, persistent=False)

        self.pose_embs = nn.Embedding(self.n_embs, emb_dim)

        # initialize refinement network
        rvec_dim = 3 if not rot_6d else 6
        N_joints = kp3d.shape[1]
        rvec_dim = rvec_dim * N_joints
        net = [nn.Linear(rvec_dim + emb_dim, width), nn.ReLU()]
        for _ in range(depth - 2):
            net.extend([nn.Linear(width, width), nn.ReLU()])
        net.append(nn.Linear(width, rvec_dim + 3))
        self.refine_net = nn.Sequential(*net)

    def forward(
        self,
        network_inputs: Dict[str, Any],
        kp3d: torch.Tensor,
        bones: torch.Tensor, # FIXME: rvec
        kp_idxs: torch.LongTensor,
        N_unique: int = 1,
        unroll: bool = True,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        kp3d: torch.Tensor (B, J, 3) -- joint locations
        bonees: torch.Tensor (B, J, 3 or 6) -- rotation vector for each joint/bone
        kp_idxs: torch.LongTensor -- index of the poses 
        unroll: bool -- unroll the for loop for kinematic calculation, this gives faster forward pass
        """
        N_samples = len(kp3d)
        skip = N_samples // N_unique
        N, J, _ = bones.shape
        # The body poses are organized in (N_images * N_rays_per_image, ...)
        # meaning that there will be N_rays_per_image redundant poses (the same image has the same pose)
        # -> skip the redundant one
        rvecs = rearrange(bones[::skip], 'b j d -> b (j d)')
        kp_unique = kp_idxs[::skip]

        embs = self.pose_embs(kp_unique)
        pelvis = self.pelvis[kp_unique].clone()
        x = torch.cat([rvecs, embs], dim=-1)
        residual = self.refine_net(x) * self.residual_scale
        residual_rvecs, residual_pelvis = torch.split(residual, [rvecs.shape[-1], 3], dim=-1)
        rvecs = rearrange(rvecs + residual_rvecs, 'b (j d) -> b j d', j=J)
        pelvis = pelvis + residual_pelvis
        rest_pose = self.rest_pose.clone()

        kp, skts = calculate_kinematic(
            rest_pose[None],
            rvecs,
            root_locs=pelvis,
            skel_type=self.skel_type,
            unroll_kinematic_chain=unroll,
        )

        # to update
        # 1. kp3d, 2. skts, 3. bones
        # -> expand the shape
        kp = rearrange(kp[:, None].expand(-1, skip, -1, -1), 'i s j d -> (i s) j d')
        skts = rearrange(skts[:, None].expand(-1, skip, -1, -1, -1), 'i s j k d -> (i s) j k d')
        rvecs = rearrange(rvecs[:, None].expand(-1, skip, -1, -1), 'i s j d -> (i s) j d')
        network_inputs.update(
            kp3d=kp,
            skts=skts,
            bones=rvecs, # FIXIT: nameing
        )
        return network_inputs
