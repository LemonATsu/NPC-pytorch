import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange

from .embedding import Optcodes
from hydra.utils import instantiate

from core.utils.ray_utils import kp_to_valid_rays
from core.utils.skeleton_utils import (
    Skeleton,
    SMPLSkeleton,
    get_skel_profile_from_rest_pose,
    calculate_kinematic,
)

from einops import rearrange
from core.networks.camcal import CamCal
from typing import Optional, Callable, Mapping, Any, List, Union
from omegaconf import DictConfig


def batchify_fn(
    fn: Callable, 
    fn_inputs: Mapping[str, Any], 
    N_total: int, 
    render_normal: bool = False, 
    chunk: int = 4096
):
    """ Break evaluation into batches to avoid OOM
    """
    all_ret = {}

    for i in range(0, N_total, chunk):
        batch_inputs = {k: fn_inputs[k][i:i+chunk] if torch.is_tensor(fn_inputs[k]) else fn_inputs[k]
                        for k in fn_inputs}
        ret = fn(batch_inputs, render_normal=render_normal)
        if ret is None:
            continue 

        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def merge_encodings(
    encoded: Mapping[str, torch.Tensor], 
    encoded_is: Mapping[str, torch.Tensor], 
    sorted_idxs: torch.Tensor,
    N_rays: int, 
    N_total_samples: int, 
    inplace: bool = True
):
    """
    merge coarse and fine encodings.
    Parameters
    ----------
    encoded: dictionary of coarse encodings
    encoded_is: dictionary of fine encodings
    sorted_idxs: define how the [encoded, encoded_is] are sorted
    """
    if encoded_is is None:
        return encoded
    if encoded is None:
        return encoded_is
    gather_idxs = torch.arange(N_rays * (N_total_samples)).view(N_rays, -1)
    gather_idxs = torch.gather(gather_idxs, 1, sorted_idxs)

    merged = encoded if inplace else {}

    for k in encoded.keys():
        #if not k.startswith(('pts', 'blend')) and k not in ['g', 'graph_feat', 'bone_logit']:
        if k in ['valid_idxs', 'rw'] or encoded[k] is None or isinstance(encoded[k], dict):
            continue
        if k in encoded and k not in encoded_is:
            merged[k] = encoded[k]
            continue
        merged[k] = merge_samples(encoded[k], encoded_is[k], gather_idxs, N_total_samples)

    # need special treatment here to preserve the computation graph.
    # (otherwise we can just re-encode everything again, but that takes extra computes)
    if 'pts' in encoded and encoded['pts'] is not None:
        if not inplace:
            merged['pts'] = encoded['pts']

        merged['pts_is'] = encoded_is['pts']
        merged['gather_idxs'] = gather_idxs

        merged['pts_sorted'] = merge_samples(encoded['pts'], encoded_is['pts'],
                                                gather_idxs, N_total_samples)

    return merged

def merge_samples(
    x: torch.Tensor, 
    x_is: torch.Tensor, 
    gather_idxs: torch.Tensor, 
    N_total_samples: int
):
    """
    merge coarse and fine samples.
    Parameteters
    ------------
    x: coarse samples of shape (N_rays, N_coarse, -1)
    x_is: importance samples of shape (N_rays, N_fine, -1)
    gather_idx: define how the [x, x_is] are sorted
    """
    if x is None or x.shape[-1] == 0:
        return None
    N_rays = x.shape[0]
    if x.shape[0] != x_is.shape[0]:
        return torch.cat([x, x_is], dim=0)
    x_is = torch.cat([x, x_is], dim=1)
    sh = x_is.shape
    feat_size = np.prod(sh[2:])
    if x_is.shape[1] != N_total_samples:
        # extra signal: may not have the same shape as the standard ray samples
        return x_is
    x_is = x_is.view(-1, feat_size)[gather_idxs, :]
    x_is = x_is.view(N_rays, N_total_samples, *sh[2:])

    return x_is

class ANeRF(nn.Module):

    def __init__(
        self,
        D: int,
        W: int,
        view_W: int,
        pts_embedder: DictConfig,
        pts_posi_enc: DictConfig,
        view_embedder: DictConfig,
        view_posi_enc: DictConfig,
        raycaster: DictConfig,
        skel_type: Skeleton,
        rest_pose: np.ndarray,
        skips: List[int] = [4],
        pred_sdf: bool = False,
        use_framecodes: bool = False,
        framecode_ch: int = 16,
        n_framecodes: int = 0,
        density_noise_std: float = 1.0,
        bkgd_net: Optional[DictConfig] = None,
        cam_cal: Optional[DictConfig] = None,
        **kwargs,
    ):
        '''
        Parameters 
        ---------- 
        D: int, depth of the MLP
        W: int, width of the MLP
        view_W: int, width of the view MLP
        pts_embedder: embedder module config, to encode 3d points w.r.t. body keypoints and poses 
        pts_posi_enc: positional encoding config for the pts embedding
        view_embedder: embedder config, to encode view vectors w.r.t. body info
        view_posi_enc: positional encoding config for the view embedding
        raycast: ray casting module config
        skel_type: skeleton, define the details of the skeleton
        skips: list, layers to do skip connection
        pred_sdf: Bool, pred_sdf=True then predicts SDF value
        use_framecodes: Bool, to use framecodes for each frame
        framecode_ch: int, size of the framecode
        n_framecodes: int, number of framecodes
        density_noise_std: float, noise to apply on density during training time
        '''
        super(ANeRF, self).__init__()

        self.D = D
        self.W = W
        self.view_W = view_W
        self.skips = skips
        self.pred_sdf = pred_sdf
        self.use_framecodes = use_framecodes
        self.framecode_ch = framecode_ch
        self.n_framecodes = n_framecodes
        self.skel_type = skel_type
        self.density_noise_std = density_noise_std

        # initialize skeleton settings
        self.init_skeleton(skel_type, rest_pose)
        # instantiate embedder and network
        self.init_embedder(
            pts_embedder=pts_embedder, 
            pts_posi_enc=pts_posi_enc, 
            view_embedder=view_embedder, 
            view_posi_enc=view_posi_enc, 
            **kwargs
        )
        self.init_raycast(raycaster, **kwargs)
        self.init_density_net()
        self.init_radiance_net()
        self.init_bkgd_net(bkgd_net)

        self.cam_cal = None
        if cam_cal is not None:
            self.cam_cal = CamCal(**cam_cal)
    
    @property
    def dnet_input(self):
        return self.input_ch
    
    @property
    def vnet_input(self):
        if self.use_framecodes:
            return self.input_ch_view + self.framecode_ch + self.W
        return self.input_ch_view + self.W
    
    def init_skeleton(self, skel_type: Skeleton, rest_pose: np.ndarray):
        self.register_buffer('rest_pose', torch.tensor(rest_pose))
        self.skel_profile = get_skel_profile_from_rest_pose(rest_pose, skel_type)
    
    def init_embedder(
        self,
        pts_embedder: DictConfig,
        pts_posi_enc: DictConfig,
        view_embedder: DictConfig,
        view_posi_enc: DictConfig,
        **kwargs,
    ):
        N_joints = len(self.skel_type.joint_names)
        # initialize points transformation
        self.pts_embedder = instantiate(pts_embedder, N_joints=N_joints, N_dims=3)
        pts_dims, pose_dims = self.pts_embedder.dims

        # initialize positional encoding for points
        self.pts_dims = pts_dims
        self.pose_dims = pose_dims
        self.pts_posi_enc = instantiate(pts_posi_enc, input_dims=pts_dims)

        # initialize view transformation
        self.view_embedder = instantiate(view_embedder, N_joints=N_joints, N_dims=3)

        # initialize positional encoding for views (rays)
        self.view_dims = view_dims= self.view_embedder.dims
        self.view_posi_enc = instantiate(view_posi_enc, input_dims=view_dims, dist_inputs=True)

        self.input_ch = self.pts_posi_enc.dims + pose_dims
        self.input_ch_view = self.view_posi_enc.dims

    def init_raycast(
        self,
        raycast: DictConfig,
        **kwargs,
    ):
        self.raycast = instantiate(raycast, **kwargs, _recursive_=False)

    def init_density_net(self):

        W, D = self.W, self.D

        layers = [nn.Linear(self.dnet_input, W)]

        for i in range(D-1):
            if i not in self.skips:
                layers += [nn.Linear(W, W)]
            else:
                layers += [nn.Linear(W + self.dnet_input, W)]

        self.pts_linears = nn.ModuleList(layers)
        self.sigma_linear = nn.Linear(W, 1)

        if self.pred_sdf:
            self.register_parameter('B',
                nn.Parameter(torch.tensor(0.1), requires_grad=True)
            )

    def init_radiance_net(self):

        W, view_W = self.W, self.view_W

        # Note: legacy code, don't really need nn.ModuleList
        self.views_linears = nn.ModuleList([nn.Linear(self.vnet_input, view_W)])
        self.feature_linear = nn.Linear(W, view_W * 2)
        self.rgb_linear = nn.Linear(view_W, 3)

        if self.use_framecodes:
            self.framecodes = Optcodes(self.n_framecodes, self.framecode_ch)
    
    def init_bkgd_net(self, bkgd_net: DictConfig):
        if bkgd_net is None:
            self.bkgd_net = None
            return 
        self.bkgd_net = instantiate(
            bkgd_net, 
            n_framecodes=self.n_framecodes, 
            framecode_ch=self.framecode_ch
        )
    
    def forward(self, *args, forward_type: str = 'rays', **kwargs):
        """ Forward function.

        Args:
            forward_type (str): ['rays', 'render', 'geometry']
        
        Returns:
            Dictionary of output tensors.
        """
        if forward_type == 'rays':
            return self.forward_rays(*args, **kwargs)
        elif forward_type == 'render':
            return self.forward_render(*args, **kwargs)
        elif forward_type =='geometry':
            return self.forward_geometry(*args, **kwargs)
        else:
            raise NotImplementedError(f'Unknown forward type {forward_type}')
    
    def forward_rays(self, batch: Mapping[str, Any], **kwargs):

        if 'rays_o' not in batch and 'rays_d' not in batch:
            raise NotImplementedError(
                'Rays are not provided as input. '
                'For rendering image with automatically detected rays, set forward_type="render"'
            )
        
        # Step 1. cast ray
        bgs = batch.get('bgs', None) # background color
        sample_info = self.raycast(batch)
        pts, z_vals = sample_info['pts'], sample_info['z_vals']

        # Step 2. model evaluation
        # TODO: do we need a get_nerf_inputs function?
        network_inputs = self.get_network_inputs(batch, pts, z_vals)
        raw, encoded = self.evaluate_pts(network_inputs, coarse=True)

        if raw is None:
            return self._empty_outputs(batch)
        
        bgnet_ret = None
        if self.training and self.bkgd_net is not None:
            bgnet_ret = self.bkgd_net(batch)
            bgs = bgnet_ret['bgs']

        # Step 3. coarse rendering
        rendered = self.raw2outputs(
            raw, 
            z_vals, 
            batch['rays_d'],
            bgs=bgs,
            encoded=encoded, 
            batch=batch,
        )

        if self.raycast.N_importance == 0:
            return {'rendered': rendered, 'encoded': encoded}

        # Step 4. importance sampling (if importance sampling enabled)
        rendered_coarse = rendered
        encoded_coarse = encoded

        weights = rendered_coarse['weights']
        is_sample_info = self.raycast(
            batch, 
            pts=pts, 
            z_vals=z_vals,
            weights=weights,
            importance=True,
        )
        pts_is = is_sample_info['pts'] # only importance samples
        z_vals = is_sample_info['z_vals'] # include both coarse and importance samples

        # Step 5. model evaluation (if importance sampling enabled)
        network_inputs = self.get_network_inputs(batch, pts_is, is_sample_info['z_vals_is'])
        raw_is, encoded_is = self.evaluate_pts(network_inputs, encoded_coarse=encoded_coarse)

        # Step 6. merge coarse and importance prediction for rendering
        sorted_idxs = is_sample_info['sorted_idxs']
        N_rays = len(batch['rays_o'])
        N_total_samples = pts.shape[1] + pts_is.shape[1]
        encoded_is = merge_encodings(encoded_coarse, encoded_is, sorted_idxs, N_rays, N_total_samples)

        # Step 7. fine rendering
        if raw_is is not None:
            raw = merge_encodings({'raw': raw}, {'raw': raw_is}, sorted_idxs, N_rays, N_total_samples)['raw']
            rendered = self.raw2outputs(
                raw, 
                z_vals, 
                batch['rays_d'],
                bgs=bgs,
                encoded=encoded_is, 
                batch=batch,
            )
        else:
            rendered = rendered_coarse

        if bgnet_ret is not None:
            rendered.update(**bgnet_ret)
        
        return self.collect_outputs(rendered, rendered_coarse, encoded_is, encoded_coarse)
    
    def forward_render(
        self, 
        render_data: Mapping[str, Any],
        render_factor: float = 0, 
        raychunk: int = 1024*10, 
        render_normal: bool = False, 
        **kwargs
    ):
        """ Render images with automatically detected rays from camera parameters.

        Args:
            render_data (Mapping[str, Any]): dictionary of rendering data, including skeleton structure and camera info
            render_factor (float): downsample factor for rendering
            raychunk (int): chunk size for ray casting
            render_normal (bool): whether to render surface normal
        
        Returns:
            Dictionary of rendered outputs.
        """

        assert 'bones' in render_data, 'needs know the pose/bones (bones) parameter during rendering'
        if 'kp3d' not in render_data:
            kp3d, skts = calculate_kinematic(
                self.rest_pose, 
                render_data['bones'],
                render_data.get('root_locs', None),
            )
            render_data['kp3d'] = kp3d
            render_data['skts'] = skts
            

        H, W, focals = render_data['hwf']
        kp3d = render_data['kp3d']
        cam_poses = render_data['c2ws'] # camera-to-world
        centers = render_data.get('center', None)

        if render_factor != 0:
            # change camera setting
            H, W = H // render_factor, W // render_factor
            focals = focals / render_factor
            if centers is not None:
                centers = centers / render_factor
            bgs = render_data['bgs']
            N, _H, _W, C = bgs.shape

            bgs = rearrange(bgs, 'n h w c -> n c h w')
            bgs = F.interpolate(bgs, scale_factor=1 / render_factor, mode='bilinear', align_corners=False)
            bgs = rearrange(bgs, 'n c h w -> n h w c')
            render_data.update(bgs=bgs)
            
        if len(cam_poses) != len(kp3d):
            assert len(kp3d) == 1 or len(cam_poses) == 1, \
                   f'Number of body poses should either match or can be broadcasted to number of camera poses. ' \
                   f'Got {len(kp3d)} and {len(cam_poses)}'

        rgb_imgs = []
        disp_imgs = []
        alpha_imgs = []

        for i in range(len(cam_poses)):

            # Step 1. find valid rays for this camera 
            center = centers[i:i+1] if centers is not None else None
            rays, valid_idxs, cyls, _ = kp_to_valid_rays(
                cam_poses[i:i+1], 
                H[i:i+1].cpu().numpy(), 
                W[i:i+1].cpu().numpy(),
                focals[i:i+1].cpu().numpy(), 
                kps=kp3d[i:i+1], 
                centers=center,
            )
            rays_o, rays_d = rays[0]
            valid_idxs = valid_idxs[0]

            # initialize images
            bg = render_data['bgs'][i].cpu()

            if valid_idxs is None or (len(valid_idxs) == 0):
                rgb_imgs.append(bg.clone())
                disp_imgs.append(torch.zeros_like(bg[..., 0]).cpu())
                continue

            # flatten to (H * W, 3)
            rgb_img = bg.clone().flatten(end_dim=1)
            disp_img = torch.zeros_like(bg[..., 0]).flatten(end_dim=1)

            # Step 2. turn them into the format that forward_rays takes
            render_data.update(cyls=cyls)
            batch = self.to_ray_inputs(rays_o, rays_d, render_data, valid_idxs, index=i)

            # Step 3. forward
            if not self.pred_sdf:
                with torch.no_grad():
                    preds = batchify_fn(self.forward_rays, batch, N_total=len(rays_o), chunk=raychunk)
            else:
                preds = batchify_fn(self.forward_rays, batch, N_total=len(rays_o), 
                                    chunk=raychunk, render_normal=render_normal)
                preds = {k: v.detach() for k, v in preds.items()}
            
            # put the rendered values into images
            pred_rgb = preds['rgb_map'].detach()
            rgb_img[valid_idxs] = pred_rgb.cpu()
            rgb_img = rgb_img.reshape(H[i], W[i], 3)

            #pred_disp = preds['disp_map'].detach()
            # TODO: hack, fix this
            pred_disp = preds['acc_map']
            disp_img[valid_idxs] = pred_disp.cpu()
            disp_img = disp_img.reshape(H[i], W[i], 1)

            rgb_imgs.append(rgb_img)
            disp_imgs.append(disp_img)

        rgb_imgs = torch.stack(rgb_imgs, dim=0)
        disp_imgs = torch.stack(disp_imgs, dim=0)

        return {
            'rgb_imgs': rgb_imgs,
            'disp_imgs': disp_imgs
        }

    def get_network_inputs(
        self, 
        batch: Mapping[str, Any],
        pts: torch.Tensor, 
        z_vals: torch.Tensor,
        keys_from_batch: List[str] = [
            'kp3d', 'skts', 'bones', 'cam_idxs', 
            'rays_o', 'rays_d', 'temp_kp', 'temp_bone',
            'temp_skt', 'real_kp_idx', 'real_cam_idx'],
        **kwargs
    ):
        """ Collect network inputs from the batch.

        Args:
            batch (Mapping[str, Any]): batch data
            pts (torch.Tensor): 3D points
            z_vals (torch.Tensor): z values of the points
            keys_from_batch (List[str]): specifying which keys to collect from the batch
        
        Returns:
            Dictionary containing network inputs.
        """
        ret = {k: batch[k] for k in keys_from_batch if k in batch}
        ret['pts'] = pts
        ret['N_unique'] = batch.get('N_unique', 1)

        if self.cam_cal is not None:
            ret = self.cam_cal(ret, z_vals)

        return ret
    
    def to_ray_inputs(
        self,
        rays_o: torch.Tensor, 
        rays_d: torch.Tensor, 
        render_data: Mapping[str, Any],
        valid_idxs: torch.Tensor, 
        index: int,
        keys_for_batch: List[str] = ['kp3d', 'skts', 'bones', 'cyls', 'cam_idxs', 'bgs'],
        pixel_data: List[str] = ['bgs'],
    ):
        """ Turning ray data into the format that forward_rays takes.

        Args:
            rays_o (torch.Tensor): ray origins
            rays_d (torch.Tensor): ray directions
            render_data (Mapping[str, Any]): dictionary of rendering data
            valid_idxs (torch.Tensor): valid ray indices
            index (int): index of the current camera
            keys_for_batch (List[str]): specifying which keys to collect from the batch
            pixel_data (List[str]): list of keys that have data in (H, W, C) format
        
        Returns:
            Dictionary containing inputs for the forward_ray function
        """

        device = rays_o.device
        ray_inputs = {
            'rays_o': rays_o,
            'rays_d': rays_d,
            'N_unique': 1, # always one pose at a time
        }

        N_rays = len(rays_o)
        # TODO: hacky to define it here..

        for k in keys_for_batch:
            v = render_data[k]
            
            if len(v) > 1:
                v = v[index:index+1]

            if k not in pixel_data:
                sh = v.shape[1:]
                v = v.expand(N_rays, *sh)
            else:
                v = rearrange(v, 'b h w c -> (b h w) c')[valid_idxs.to(v.device)].to(device)
            ray_inputs[k] = v
        return ray_inputs

    def evaluate_pts(self, inputs: Mapping[str, Any], geometry_only: bool = False, geom_extra_rets: List[str] = [], **kwargs):
        """ Evaluate the NeRF function at the given inputs/points.
        """

        if self.pred_sdf:
            inputs['pts'].requires_grad = True

        # Step 1. encode all pts feature
        density_inputs, encoded_pts = self.encode_pts(inputs)
        if density_inputs is None:
            # terminate because not valid points 
            return None, None
        
        # Step 2. density prediction
        sigma, density_feature = self.inference_sigma(density_inputs)

        if self.pred_sdf:
            surface_gradient = torch.autograd.grad(
                outputs=sigma,
                inputs=inputs['pts'],
                grad_outputs=torch.ones_like(sigma),
                create_graph=self.training,
            )[0]
            encoded_pts.update(surface_gradient=surface_gradient)

        density = self.to_density(sigma)

        if geometry_only:
            geom_ret = {'density': density, 'sigma': sigma}
            if 'valid_idxs' in encoded_pts:
                geom_ret['valid_idxs'] = encoded_pts['valid_idxs']
            if 'surface_gradient' in encoded_pts:
                geom_ret['surface_gradient'] = encoded_pts['surface_gradient']
            for k in geom_extra_rets:
                if k in encoded_pts:
                    geom_ret[k] = encoded_pts[k]
                if k == 'density_feature':
                    geom_ret[k] = density_feature
            return None, geom_ret

        # Step 3. encode all ray feature
        view_inputs, encoded_views = self.encode_views(
            inputs, 
            refs=encoded_pts['pts_t'],
            encoded_pts=encoded_pts
        )
        
        # Step 4. rgb prediction
        rgb = self.inference_rgb(view_inputs, density_feature)

        # Step 5: create final outputs
        shape = inputs['pts'].shape[:2] # (N_rays, N_samples)
        outputs = self.fill_prediction(torch.cat([rgb, density], dim=-1), shape, encoded_pts)

        return outputs, self.collect_encoded(encoded_pts, encoded_views)

    def encode_pts(self, inputs: Mapping[str, Any]):
        """ Encode input points into encoding needed for the NeRF model.

        Args:
            inputs (Mapping[str, Any]): dictionary of inputs
        
        Returns:
            density_inputs (torch.Tensor): density network inputs
            encoded (Mapping[str, torch.Tensor]): detail encoding of the points
        """

        N_rays, N_samples = inputs['pts'].shape[:2]

        encoded = self.pts_embedder(**inputs)
        v_pe = self.pts_posi_enc(encoded['v'])[0]
        r = encoded['r'].reshape(N_rays, N_samples, self.pose_dims)

        # apply positional encoding (PE)
        # pe_fn returns a tuple: (encoded outputs, cutoff weights)
        density_inputs = torch.cat([v_pe, r], dim=-1).flatten(end_dim=1)

        return density_inputs, encoded

    def encode_views(self, inputs: Mapping[str, Any], refs: torch.Tensor, encoded_pts: Mapping[str, torch.Tensor]):
        """ Encode input views into encoding needed for the NeRF model.

        Args:
            inputs (Mapping[str, Any]): dictionary of inputs
            refs (torch.Tensor): reference tensor for expanding rays
            encoded_pts (Mapping[str, Any]): point encoding that could be useful for encoding view
        
        Returns:
            view_inputs (torch.Tensor): view network inputs
            encoded (Mapping[str, Any]): detail encoding of the views
        """

        N_rays, N_samples = refs.shape[:2]
        encoded = self.view_embedder(**inputs, refs=refs)
        d = encoded['d'].reshape(N_rays, N_samples, self.view_dims)
        
        # apply positional encoding (PE)
        d_pe = self.view_posi_enc(d, dists=encoded_pts.get('v', None))[0]

        view_inputs = d_pe
        if self.use_framecodes:
            N_rays, N_samples = refs.shape[:2]
            # expand from (N_rays, ...) to (N_rays, N_samples, ...)
            cam_idxs = inputs['cam_idxs']
            cam_idxs = cam_idxs.reshape(N_rays, 1, -1).expand(-1, N_samples, -1)
            framecodes = self.framecodes(cam_idxs.reshape(N_rays * N_samples, -1))
            framecodes = framecodes.reshape(N_rays, N_samples, -1)
            view_inputs = torch.cat([view_inputs, framecodes], dim=-1)

        if 'normal' in encoded_pts and self.pred_sdf:
            normal = encoded_pts['normal']
            view_inputs = torch.cat([view_inputs, normal], dim=-1)

        view_inputs = view_inputs.flatten(end_dim=1)
        if 'valid_idxs' in encoded_pts:
            view_inputs = view_inputs[encoded_pts['valid_idxs']]

        return view_inputs, encoded

    def fill_prediction(self, preds: torch.Tensor, shape: Union[List, tuple], valid_info: Mapping[str, Any]):
        """ Create a full tensor from valid prediction.
        In evaluate_pts, we may avoid computation on some points that do not require prediction.
        The preds tensor shape is thus varying. This function turn the prediction into fixed size
        so that the prediction can be processed with batch operation.

        Args:
            preds (torch.Tensor): prediction tensor
            shape (Union[List, tuple]): shape of the first dimension in preds
            valid_info (Mapping[str, Any]): information that requires to map preds back to a tensor of shape [*shape, pred_size]
        
        Returns:
            outputs (torch.Tensor): tensor of shape [*shape, pred_size]
        """
        output_dim = preds.shape[-1]

        if valid_info is not None and 'valid_idxs' in valid_info:
            valid_idxs = valid_info['valid_idxs']
            outputs = torch.zeros(np.prod(shape), output_dim)
            # by default force nothing there in the empty space

            # outputs[invalid_idxs, -1] = 0.
            outputs[valid_idxs] += preds
            outputs = outputs.reshape(*shape, output_dim)
        else:
            outputs = preds.reshape(*shape, output_dim) 

        return outputs
    
    def inference_sigma(self, density_inputs: torch.Tensor):
        """ Inference the geometry representation sigma (density or sdf) values
        """
        h = self.forward_density(density_inputs)
        sigma = self.sigma_linear(h)
        return sigma, h
    
    def inference_rgb(self, view_inputs: torch.Tensor, density_feature: torch.Tensor, rgb_eps: float = 0.001, **kwargs):
        """ Inference the color for the given input points
        """
        rgb = self.forward_view(view_inputs, density_feature)
        rgb = torch.sigmoid(rgb) * (1 + 2 * rgb_eps) - rgb_eps 
        return rgb

    def forward_density(self, density_inputs: torch.Tensor):
        h = density_inputs

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h, inplace=True)
            if i in self.skips:
                h = torch.cat([density_inputs, h], -1)
        return h

    def forward_view(self, view_inputs: torch.Tensor, density_feature: torch.Tensor):
        # produce features for color/radiance
        feature = self.feature_linear(density_feature)
        h = torch.cat([feature, view_inputs], -1)

        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h, inplace=True)

        return self.rgb_linear(h)
    
    def forward_geometry(
        self, 
        render_data: Mapping[str, Any],
        chunk: int = 1024*64*5, 
        return_gradient: bool = False, 
        geom_extra_rets: List[str] = [], 
        **kwargs
    ):
        """ Forward function for geometry only.

        Args:
            render_data (Mapping[str, Any]): dictionary of rendering data
            chunk (int): chunk size for forward to avoid OOM
            return_gradient (bool): whether to return gradient
            geom_extra_rets (List[str]): list of keys of extra tensors to return
        
        Returns:
            Dictionary of outputs.
        """

        assert 'pts' in render_data, f'Query locations need to be provided for forward_geometry'
        if 'kp3d' not in render_data:
            assert self.skel_type == SMPLSkeleton, f'Only SMPLSkeleton is supported for kinematic forward!'
            kp3d, skts = calculate_kinematic(
                self.rest_pose, 
                render_data['bones'],
                render_data.get('root_locs', None),
            )
            render_data['skts'] = skts
            render_data['kp3d'] = kp3d
        
        pts = render_data['pts']
        kp3d = render_data['kp3d']
        skts = render_data['skts']
        bones = render_data['bones']

        # only deals with one pose at a time
        assert len(kp3d) == 1
        assert len(skts) == 1
        assert len(bones) == 1

        pts_shape = pts.shape
        pts = pts.reshape(1, -1, 3)
        N_samples = pts.shape[1]

        # pre-allocate
        density = torch.zeros(N_samples, 1)
        sigma = torch.zeros(N_samples, 1)
        extra_dict = {k: None for k in geom_extra_rets}
        if self.pred_sdf:
            sigma += 10.
        gradient = None
        if return_gradient:
            gradient = torch.zeros(N_samples, 3)
        for i in range(0, N_samples, chunk):
            chunk_pts = pts[:, i:i+chunk]
            geom_inputs = {
                'kp3d': kp3d,
                'skts': skts,
                'bones': bones,
                'N_unique': 1,
                'pts': chunk_pts,
            }
            _, preds = self.evaluate_pts(
                geom_inputs, 
                geometry_only=True, 
                geom_extra_rets=geom_extra_rets,
            )
            if preds is None:
                continue
            preds = {k: v.detach() for k, v in preds.items()}
            if 'valid_idxs' in preds:
                density[i + preds['valid_idxs']] = preds['density']
                sigma[i + preds['valid_idxs']] = preds['sigma']
                if gradient is not None:
                    
                    pred_gradient = preds['surface_gradient'].flatten(end_dim=1)
                    pred_gradient = pred_gradient[preds['valid_idxs']]
                    gradient[i + preds['valid_idxs']] = pred_gradient
                
                for k in geom_extra_rets:
                    extra_tensor = preds[k]
                    if extra_dict[k] is None:
                        extra_dict[k] = torch.zeros(N_samples, extra_tensor.shape[-1])
                    extra_dict[k][i + preds['valid_idxs']] = extra_tensor
            else:
                density[i:i+chunk] = preds['density']
                sigma[i:i+chunk] = preds['sigma']
                if gradient is not None:
                    gradient[i:i+chunk] = preds['surface_gradient'].flatten(end_dim=1)

                for k in geom_extra_rets:
                    extra_tensor = preds[k]
                    if extra_dict[k] is None:
                        extra_dict[k] = torch.zeros(N_samples, extra_tensor.shape[-1])
                    extra_dict[k][i:i+chunk] = extra_tensor
        
        outputs = {
            'density': density.reshape(*pts_shape[:-1], 1),
            'sigma': sigma.reshape(*pts_shape[:-1], 1),
            **extra_dict,
        }


        if gradient is not None:
            outputs.update(gradient=gradient.reshape(*pts_shape[:-1], 3))
        return outputs

    def collect_encoded(self, encoded_pts: Mapping[str, torch.Tensor], encoded_views: Mapping[str, torch.Tensor]):
        """ Collect encodings that are useful for loss computations or rendering
        """
        ret = {}
        if self.pred_sdf and 'surface_gradient' in encoded_pts:
            ret['surface_gradient'] = encoded_pts['surface_gradient']
        return ret

    def raw2outputs(
        self, 
        raw: torch.Tensor, 
        z_vals: torch.Tensor, 
        rays_d: torch.Tensor, 
        bgs: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # TODO: update
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
            bgs: [num_rays, 3]. Background color
            act_fn: activation function for the density
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """
        if self.pred_sdf:
            return self.raw2sdfoutputs(raw, z_vals, rays_d, bgs=bgs, **kwargs)

        #raw2alpha = lambda raw, dists, noise, act_fn=act_fn: 1.-torch.exp(-(act_fn(raw + noise))*dists)

        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        rgb = raw[..., :3]

        alpha = 1. - torch.exp(-raw[..., 3] * dists) 

        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        # sum_{i=1 to N samples} prob_of_already_hit_particles * alpha_for_i * color_for_i
        # C(r) = sum [T_i * (1 - exp(-sigma_i * delta_i)) * c_i] = sum [T_i * alpha_i * c_i]
        # alpha_i = 1 - exp(-sigma_i * delta_i)
        # T_i = exp(sum_{j=1 to i-1} -sigma_j * delta_j) = torch.cumprod(1 - alpha_i)
        # standard NeRF
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / (torch.sum(weights, -1)  + 1e-10))

        invalid_mask = torch.ones_like(disp_map)
        invalid_mask[torch.isclose(weights.sum(-1), torch.tensor(0.))] = 0.
        disp_map = disp_map * invalid_mask

        acc_map = torch.minimum(torch.sum(weights, -1), torch.tensor(1.))

        if bgs is not None:
            rgb_map = rgb_map + (1. - acc_map)[..., None] * bgs
        
        return {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map,
                'weights': weights, 'alpha': alpha}

    def raw2sdfoutputs(
        self, 
        raw: torch.Tensor, 
        z_vals: torch.Tensor, 
        rays_d: torch.Tensor, 
        bgs: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """ Sligtly different from usual case
        """

        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        # following Lior's paper
        density = raw[..., 3]
        # note: this is in logspace, originally we do exp(-density*dists)
        free_energy = density * dists
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1), free_energy[:, :-1]], dim=-1)  # shift one step

        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        # sum_{i=1 to N samples} prob_of_already_hit_particles * alpha_for_i * color_for_i
        # C(r) = sum [T_i * (1 - exp(-sigma_i * delta_i)) * c_i] = sum [T_i * alpha_i * c_i]
        # alpha_i = 1 - exp(-sigma_i * delta_i)
        # T_i = exp(sum_{j=1 to i-1} -sigma_j * delta_j) = torch.cumprod(1 - alpha_i)
        # standard NeRF
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        T = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * T # probability of the ray hits something here
        #weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]

        # get rgb (as usual)
        rgb = raw[..., :3] # [N_rays, N_samples, 3]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        eps = 1e-10
        inv_eps = 1. / eps
        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = (weights.sum(dim=-1) / depth_map).double()
        disp_map = torch.where(
            (disp_map > 0) & (disp_map < inv_eps) & (weights.sum(dim=-1) > eps), disp_map, inv_eps
        ).float()
        encoded = kwargs['encoded']
        """
        disp_map = 1./torch.max(eps * torch.ones_like(depth_map), depth_map / (torch.sum(weights, -1)  + eps))

        invalid_mask = torch.ones_like(disp_map)
        invalid_mask[torch.isclose(weights.sum(-1), torch.tensor(0.))] = 0.
        disp_map = disp_map * invalid_mask
        """

        acc_map = torch.minimum(torch.sum(weights, -1), torch.tensor(1.))

        if bgs is not None:
            rgb_map = rgb_map + (1. - acc_map)[..., None] * bgs

        return {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map,
                'weights': weights, 'alpha': alpha}

    def collect_outputs(
        self, 
        ret: Mapping[str, torch.Tensor], 
        ret0: Optional[Mapping[str, torch.Tensor]] = None, 
        encoded: Optional[Mapping[str, torch.Tensor]] = None, 
        encoded0: Optional[Mapping[str, torch.Tensor]] = None
    ):
        """ Collect outputs into a dictionary for loss computation/rendering
        
        Parameter
        ---------
        ret: dictionary of the fine-level rendering
        ret0: dictionary of the coarse-level rendering
        encoded: dictionary of the fine-level model info
        encoded0: dictionary of the corase-level model info

        """

        collected = {'rgb_map': ret['rgb_map'], 'disp_map': ret['disp_map'],
                     'acc_map': ret['acc_map'],}
        if not self.training:
            return collected

        collected['T_i'] = ret['weights']
        collected['alpha'] = ret['alpha']
        
        if ret0 is not None:
            collected['rgb0'] = ret0['rgb_map']
            collected['disp0'] = ret0['disp_map']
            collected['acc0'] = ret0['acc_map']
            collected['alpha0'] = ret0['alpha']

        if 'j_dists' in ret and self.training:
            collected['j_dists'] = ret['j_dists']
            if ret0 is not None:
                collected['j_dists0'] = ret0['j_dists']

        if encoded is not None and 'surface_gradient' in encoded and self.training:
            collected['surface_gradient'] = encoded['surface_gradient']
        
            if encoded0 is not None:
                collected['surface_gradient0'] = encoded0['surface_gradient']
        
        if self.training and 'bgs' in ret:
            collected['bgs'] = ret['bgs']
            collected['bg_preds'] = ret['bg_preds']

        return collected

    def _empty_outputs(self, batch: Mapping[str, Any]):
        N_rays = len(batch['rays_o'])
        empty = torch.zeros(N_rays, 3)
        rgb_empty = batch.get('bgs', empty)
        return {'rgb_map': rgb_empty, 'disp_map': empty[:, 0], 'acc_map': empty[:, 0]}
    
    def to_density(self, sigma: torch.Tensor):
        """ Turn sigma from raw prediction (value unbounded) to density (value >= 0.)
        """
        if self.pred_sdf:
            return self.sdf2density(sigma)

        noise = 0.
        if self.training and self.density_noise_std > 0.:
            noise = torch.randn_like(sigma) * self.density_noise_std

        sigma = sigma + noise
        
        return F.relu(sigma, inplace=True)

    def sdf2density(self, sdf: torch.Tensor):
        # use laplace CDF to represent density, with sdf being pdf
        B = (self.B.abs() + 0.0001)
        inv_B = 1 / B
        return inv_B * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / B))