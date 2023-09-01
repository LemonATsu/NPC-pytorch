import torch
import numpy as np
import os
import glob
import shutil
import cv2
import imageio
import lpips
from cleanfid import fid # TODO: hacked the code so it will find inception in the current folder
from skimage.metrics import structural_similarity as compare_ssim
#torch.manual_seed(12345)
#np.random.seed(12345)
torch.use_deterministic_algorithms(True)


def move_images_to_folder(src_folder, dst_folder, gt_folder, method_name=''):
    metadata = np.load(os.path.join(gt_folder, 'metadata.npy'), allow_pickle=True).item()
    num_nv = len(metadata['nv_gts'])
    num_np = len(metadata['np_gts'])
    print(f'num_nv {num_nv}, num_np {num_np}')

    img_paths = sorted(glob.glob(os.path.join(src_folder, '*.png')))
    assert len(img_paths), "image not found"
    nv_paths = img_paths[:num_nv]
    np_paths = img_paths[num_nv:]

    save_folder = os.path.join(dst_folder, method_name)
    nv_folder = os.path.join(save_folder, 'novel_view')
    np_folder = os.path.join(save_folder, 'novel_pose')
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(nv_folder, exist_ok=True)
    os.makedirs(np_folder, exist_ok=True)
    for j, nv_path in enumerate(nv_paths):
        shutil.copy(nv_path, os.path.join(nv_folder, f'{j:05d}.png'))

    for j, np_path in enumerate(np_paths):
        shutil.copy(np_path, os.path.join(np_folder, f'{j:05d}.png'))

def create_cropped_images(method_folder, gt_folder):
    metadata = np.load(os.path.join(gt_folder, 'metadata.npy'), allow_pickle=True).item()
    num_nv = len(metadata['nv_gts'])
    num_np = len(metadata['np_gts'])
    print(f'num_nv {num_nv}, num_np {num_np}')

    nv_folder = os.path.join(method_folder, 'novel_view')
    np_folder = os.path.join(method_folder, 'novel_pose')
    nv_crop_folder = os.path.join(method_folder, 'novel_view_cropped')
    np_crop_folder = os.path.join(method_folder, 'novel_pose_cropped')
    os.makedirs(nv_crop_folder, exist_ok=True)
    os.makedirs(np_crop_folder, exist_ok=True)

    full_bbox = metadata['bboxes']
    
    num_novel_views = len(metadata['nv_gts'])
    num_novel_poses = len(metadata['np_gts'])
    nv_bboxes = full_bbox[:num_novel_views]
    np_bboxes = full_bbox[num_novel_views:]
    for j, img_path in enumerate(sorted(glob.glob(os.path.join(nv_folder, '*.png')))):
        img = imageio.imread(img_path)
        x, y, w, h = nv_bboxes[j]
        x, y, w, h = int(x), int(y), int(w), int(h)
        cropped = img[y:y+h, x:x+w].copy()
        imageio.imwrite(os.path.join(nv_crop_folder, f'{j:05d}.png'), cropped)
    
    for j, img_path in enumerate(sorted(glob.glob(os.path.join(np_folder, '*.png')))):
        img = imageio.imread(img_path)
        x, y, w, h = np_bboxes[j]
        x, y, w, h = int(x), int(y), int(w), int(h)
        cropped = img[y:y+h, x:x+w].copy()
        imageio.imwrite(os.path.join(np_crop_folder, f'{j:05d}.png'), cropped)

def compute_psnr(render, gt):
    mse = np.mean((render - gt)**2)
    return -10 * np.log(mse) / np.log(10)

def evaluate_psnrs(renders, gts, masks=None):
    
    assert len(renders) == len(gts)
    if masks is not None:
        assert len(masks) == len(renders)
        

    psnrs = []
    for i, (render, gt) in enumerate(zip(renders, gts)):
        render = render.copy() 
        gt = gt.copy() 
        if masks is not None:
            psnr = compute_psnr(render[masks[i] > 0], gt[masks[i] > 0])
        else:
            psnr = compute_psnr(render, gt)
        psnrs.append(psnr)
    return np.array(psnrs)

def evaluate_ssims(renders, gts, masks=None):
    assert len(renders) == len(gts)
    if masks is not None:
        assert len(masks) == len(renders)
    ssims = []
    for i, (render, gt) in enumerate(zip(renders, gts)):
        render = render.copy() 
        gt = gt.copy() 
        if masks is not None:
            x, y, w, h = cv2.boundingRect(masks[i] * 255)

            render = render[y:y + h, x:x + w].copy()
            gt = gt[y:y + h, x:x + w].copy()


        ssims.append(compare_ssim(render, gt, channel_axis=-1, data_range=1))
    return np.array(ssims)

def evaluate_lpips(renders, gts, masks):
    loss_fn = lpips.LPIPS(net='vgg').cuda()
    loss_alex_fn = lpips.LPIPS(net='alex').cuda()
    assert len(renders) == len(gts)
    assert len(renders) == len(masks)
    lpips_vals, lpips_alex_vals = [], []
    for i, (render, gt, mask) in enumerate(zip(renders, gts, masks)):
        render = render.copy() 
        gt = gt.copy() 
        """
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        """
        x, y, w, h = cv2.boundingRect(mask * 255)

        render = render[y:y + h, x:x + w].copy()
        gt = gt[y:y + h, x:x + w].copy()
        H, W, C = gt.shape

        render_tensor = torch.tensor(render).reshape(1, H, W, C).permute(0, 3, 1, 2).float().cuda()
        gt_tensor = torch.tensor(gt).reshape(1, H, W, C).permute(0, 3, 1, 2).float().cuda()
        # to [-1, 1]
        render_tensor = render_tensor * 2 - 1.
        gt_tensor = gt_tensor * 2 - 1.
        with torch.no_grad():
            d = loss_fn(gt_tensor, render_tensor).item()
            d_alex = loss_alex_fn(gt_tensor, render_tensor).item()
        lpips_vals.append(d)
        lpips_alex_vals.append(d_alex)

    return np.array(lpips_vals), np.array(lpips_alex_vals)

def evaluate_kid_fid(render_folder, gt_folder, masks=None, n_avg=5):
    fid_score = fid.compute_fid(gt_folder, render_folder)

    kid_score_ = []
    for i in range(n_avg):
        kid_score = fid.compute_kid(gt_folder, render_folder)
        kid_score_.append(kid_score)
    kid_score = np.mean(kid_score_)
    return kid_score, fid_score


def evaluate_h36m_kid_fid_subject(data_dir, methods, subject, gt_tag='eval', baseline_tag='baselines'):
    metrics = {k: {} for k in methods}
    gt_base = os.path.join(data_dir, f'{subject}_{gt_tag}')
    method_base = os.path.join(data_dir, f'{subject}_{baseline_tag}')

    #subsets = ['novel_view_cropped', 'novel_pose_cropped']
    subsets = ['novel_pose_cropped']
    metadata = np.load(os.path.join(gt_base, f'metadata.npy'), allow_pickle=True).item()
    full_bbox = metadata['bboxes']
    
    gt_img_folder = os.path.join(gt_base, 'cropped')
    for subset in subsets:
        for method in methods:
            method_img_folder = os.path.join(method_base, method, subset)
            kid_score, fid_score = evaluate_kid_fid(method_img_folder, gt_img_folder)
            metrics[method][subset] = {'kid': [kid_score], 'fid': [fid_score]}
            print(f'{method}-{subset}: {metrics[method][subset]}')
    return metrics

def evaluate_h36m_subject(data_dir, methods, subject, gt_tag='eval', baseline_tag='baselines'):
    metrics = {k: {} for k in methods}
    gt_base = os.path.join(data_dir, f'{subject}_{gt_tag}')
    method_base = os.path.join(data_dir, f'{subject}_{baseline_tag}')

    #subsets = ['novel_view', 'novel_pose']
    subsets = ['novel_pose']
    metadata = np.load(os.path.join(gt_base, f'metadata.npy'), allow_pickle=True).item()
    full_bbox = metadata['bboxes']
    
    num_novel_views = len(metadata['nv_gts'])
    num_novel_poses = len(metadata['np_gts'])
    subset_bboxes = [full_bbox[:num_novel_views], full_bbox[num_novel_views:]]

    for bboxes, subset in zip(subset_bboxes, subsets):
        gt_mask_paths = sorted(glob.glob(os.path.join(gt_base, subset, '*_mask.png')))
        gt_img_paths = sorted([f for f in glob.glob(os.path.join(gt_base, subset, '*.png'))
                               if 'mask' not in f])
        gt_images = np.array([imageio.imread(p) for p in gt_img_paths]).astype(np.float32) / 255.
        gt_masks = np.array([imageio.imread(p) for p in gt_mask_paths]).astype(np.float32) / 255.

        # the images and masks are (1002, 1000), set them to (1000, 1000)
        gt_images = gt_images[:, 1:-1]
        gt_masks = gt_masks[:, 1:-1]

        for method in methods:
            method_folder = os.path.join(method_base, method, subset)
            img_paths = sorted(glob.glob(os.path.join(method_folder, '*.png')))
            try:
                assert len(img_paths) == len(gt_img_paths)
            except:
                print(f'method {method}')
                print
            method_images = np.array([imageio.imread(p) for p in img_paths]).astype(np.float32) / 255.

            try:
                if method_images.shape[1] == 1002:
                    method_images = method_images[:, 1:-1]
            except:
                import pdb; pdb.set_trace()
                print
            psnrs = evaluate_psnrs(method_images, gt_images, masks=gt_masks)
            ssims = evaluate_ssims(method_images, gt_images, masks=gt_masks[..., None].astype(np.uint8))
            lpips, lpips_alex = evaluate_lpips(method_images, gt_images, masks=gt_masks[..., None].astype(np.uint8))
            metrics[method][subset] = {'psnr': psnrs, 'ssim': ssims, 'lpips': lpips, 'lpips_alex': lpips_alex}
    return metrics

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Arguments for evaluation')
    parser.add_argument('-s', '--subject', type=str, default='S1', 
                        choices=['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11'], 
                        help='subject id')
    parser.add_argument('-d', '--data_dir', type=str, default='data/h36m_zju/', 
                        help='ground truth data directory')
    parser.add_argument('-i', '--image_dir', type=str, required=True, 
                        help='image directory for evaluation')
    parser.add_argument('-o', '--output_dir', type=str, default='eval_results/h36m_zju/',
                        help='location to save the evaluation results')
    parser.add_argument('-m', '--method', type=str, required=True,
                        help='method name')
    parser.add_argument('--skip_crop', action='store_true', default=False,
                        help='skip moving and cropping images if we already done that.')
    args = parser.parse_args()

    subject = args.subject
    method_name = args.method
    output_dir = args.output_dir
    image_dir = args.image_dir
    data_dir = args.data_dir
    skip_crop = args.skip_crop

    print(f'Evaluating for {method_name}')
    if not skip_crop:
        # step 1.: move the output to the target director
        dst_dir = os.path.join(data_dir, f'{subject}_baselines')
        src_dir = image_dir
        gt_dir = os.path.join(data_dir, f'{subject}_eval')
        print(f'copying images from {src_dir} to {dst_dir}, gt from {gt_dir}')
        move_images_to_folder(src_dir, dst_dir, gt_dir, method_name=method_name)

        # step 2. crop images and store
        method_dir = os.path.join(dst_dir, method_name)
        create_cropped_images(method_dir, gt_dir)
    
    # step 3. evaluate
    eval_results = evaluate_h36m_subject(data_dir, [method_name], subject=subject)
    perceptual_results = evaluate_h36m_kid_fid_subject(data_dir, [method_name], subject=subject)
    eval_results[method_name].update(**perceptual_results[method_name])

    save_dir = os.path.join(output_dir, method_name)
    save_path = os.path.join(save_dir, f'{subject}.npy')
    os.makedirs(save_dir, exist_ok=True)
    np.save(save_path, eval_results, allow_pickle=True)
    print(f'Results saved to {save_path}')

