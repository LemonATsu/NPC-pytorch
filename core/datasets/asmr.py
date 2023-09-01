import numpy as np

import h5py

from core.utils.skeleton_utils import *
from core.utils.visualization import *
from core.networks.misc import *

from core.datasets import BaseH5Dataset


class ASMRDataset(BaseH5Dataset):
    render_skip = 50

    def init_meta(self):
        super().init_meta()
        dataset = h5py.File(self.h5_path, 'r')
        self.kp_idxs = dataset['kp_idxs'][:]
        self.cam_idxs = dataset['img_pose_indices'][:]
        dataset.close()
        self.skel_type = MixamoSkeleton

        self.has_bg = True
        self.bgs = np.zeros((1, np.prod(self.HW), 3), dtype=np.uint8)
        self.bg_idxs = np.zeros((len(self.kp_idxs),), dtype=np.int64)


    def init_len(self):
        if self._idx_map is not None:
            self.data_len = len(self._idx_map)
        else:
            with h5py.File(self.h5_path, 'r') as f:
                self.data_len = len(f['imgs']) 

    def get_kp_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        return self.kp_idxs[idx], q_idx

    def get_cam_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        return self.cam_idxs[idx], q_idx

    def get_meta(self):
        '''
        return metadata needed for other parts of the code.
        '''

        data_attrs = super().get_meta()
        dataset = h5py.File(self.h5_path, 'r')
        rest_heads = dataset['rest_heads'][:]
        dataset.close()

        data_attrs['rest_heads'] = rest_heads
        data_attrs['skel_type'] = MixamoSkeleton

        return data_attrs

    def get_render_data(self):
        #h5_path = self.h5_path.replace('train', 'test')
        h5_path = self.h5_path

        H, W = self.HW
        render_skip = self.render_skip
        dataset = h5py.File(h5_path, 'r')
        render_imgs = dataset['imgs'][::render_skip].reshape(-1, H, W, 3).astype(np.float32) / 255.
        render_fgs = dataset['masks'][::render_skip].reshape(-1, H, W, 1).astype(np.float32)
        render_bgs = np.zeros_like(render_imgs[:1]).astype(np.float32)
        render_bg_idxs = np.zeros(len(render_imgs)).astype(np.int64)

        kp3d = dataset['kp3d'][::render_skip]
        skts = dataset['skts'][::render_skip]
        bones = dataset['bones'][::render_skip]
        center = dataset['centers'][::render_skip]
        focals = dataset['focals'][::render_skip]
        c2ws = dataset['c2ws'][::render_skip]

        c_idxs = dataset['img_pose_indices'][::render_skip]

        H = np.repeat([H], len(c_idxs), 0)
        W = np.repeat([W], len(c_idxs), 0)
        hwf = (H, W, focals)

        dataset.close()

        render_data = {
            'imgs': render_imgs, 
            'fgs': render_fgs,
            'bgs': render_bgs,
            'bg_idxs': render_bg_idxs,
            'bg_idxs_len': len(render_bg_idxs),
            # camera data
            'cam_idxs': c_idxs * 0 - 1, # set to -1 to use avg framecode
            'cam_idxs_len': len(c2ws),
            'c2ws': c2ws,
            'hwf': hwf,
            'center': center,
            'kp_idxs': np.arange(len(kp3d)),
            'kp_idxs_len': len(kp3d),
            'kp3d': kp3d,
            'skts': skts,
            'bones': bones,
        }
        return render_data
