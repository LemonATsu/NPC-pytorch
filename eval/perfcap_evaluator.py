import torch
import numpy as np
import os 
import glob
import shutil
import cv2
import imageio
import lpips
torch.manual_seed(12345)
np.random.seed(12345)
torch.use_deterministic_algorithms(True)

from h36m_evaluator import (
    evaluate_psnrs,
    evaluate_ssims,
    evaluate_lpips,
    evaluate_kid_fid,
)

def move_images_to_folder(src_folder, dst_folder, gt_folder, method_name=''):

    img_paths = sorted(glob.glob(os.path.join(src_folder, '*.png')))
    assert len(img_paths), "image not found"
    save_folder = os.path.join(dst_folder, method_name, 'novel_pose')
    os.makedirs(save_folder, exist_ok=True)

    for j, img_path in enumerate(img_paths):
        shutil.copy(img_path, os.path.join(save_folder, f'{j:05d}.png'))

def create_cropped_images(method_folder, gt_folder):
    bboxes = np.load(os.path.join(gt_folder, f'bboxes.npy'))
    crop_folder = os.path.join(method_folder, 'novel_pose_cropped')

    img_folder = os.path.join(method_folder, 'novel_pose')
    os.makedirs(crop_folder, exist_ok=True)

    for j, img_path in enumerate(sorted(glob.glob(os.path.join(img_folder, '*.png')))):
        bbox = bboxes[j].copy()
        tl, br = bbox
        img = imageio.imread(img_path)
        cropped_img = img[tl[1]:br[1], tl[0]:br[0]]
        print(img.shape)
        imageio.imwrite(os.path.join(crop_folder, f'{j:05d}.png'), cropped_img)

def evaluate_perfcap_kid_fid_subject(data_dir, methods, subject, gt_tag='outdoor_eval', baseline_tag='outdoor_baselines'):
    metrics = {k: {} for k in methods}
    gt_base = os.path.join(data_dir, f'{subject}_{gt_tag}')
    method_base = os.path.join(data_dir, f'{subject}_{baseline_tag}')

    subsets = ['novel_pose_cropped']
    
    gt_img_folder = os.path.join(gt_base, 'cropped')
    for subset in subsets:
        for method in methods:
            method_img_folder = os.path.join(method_base, method, subset)
            kid_score, fid_score = evaluate_kid_fid(method_img_folder, gt_img_folder)
            metrics[method][subset] = {'kid': [kid_score], 'fid': [fid_score]}
            print(f'{method}-{subset}: {metrics[method][subset]}')
    return metrics

def evaluate_perfcap_subject(
    data_dir, 
    methods, 
    subject, 
    gt_tag='outdoor_eval', 
    baseline_tag='outdoor_baselines'
):
    metrics = {k: {} for k in methods}
    gt_base = os.path.join(data_dir, f'{subject}_{gt_tag}')
    method_base = os.path.join(data_dir, f'{subject}_{baseline_tag}')

    bboxes = np.load(os.path.join(gt_base, f'bboxes.npy'))

    subsets = ['novel_pose']

    for subset in subsets:
        gt_img_paths = sorted([f for f in glob.glob(os.path.join(gt_base, subset, '*.png'))
                               if 'mask' not in f])
        
        gt_images = np.array([imageio.imread(p) for p in gt_img_paths]).astype(np.float32) / 255.
        gt_masks = np.zeros_like(gt_images[..., 0])
        for i, bbox in enumerate(bboxes):
            tl, br = bbox
            gt_masks[i, tl[1]:br[1], tl[0]:br[0]] = 1.
        
        for method in methods:
            method_folder = os.path.join(method_base, method, subset)
            img_paths = sorted(glob.glob(os.path.join(method_folder, '*.png')))
            method_images = np.array([imageio.imread(p) for p in img_paths]).astype(np.float32) / 255.

            psnrs = evaluate_psnrs(method_images, gt_images, masks=gt_masks)
            ssims = evaluate_ssims(method_images, gt_images, masks=gt_masks[..., None].astype(np.uint8))
            lpips, lpips_alex = evaluate_lpips(method_images, gt_images, masks=gt_masks[..., None].astype(np.uint8))
            metrics[method][subset] = {'psnr': psnrs, 'ssim': ssims, 'lpips': lpips, 'lpips_alex': lpips_alex}
    return metrics

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Arguments for evaluation')
    parser.add_argument('-s', '--subject', type=str, default='weipeng', 
                        choices=['weipeng', 'nadia'], 
                        help='subject id')
    parser.add_argument('-d', '--data_dir', type=str, default='data/MonoPerfCap/', 
                        help='ground truth data directory')
    parser.add_argument('-i', '--image_dir', type=str, required=True, 
                        help='image directory for evaluation')
    parser.add_argument('-o', '--output_dir', type=str, default='eval_results/perfcap/',
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
        dst_dir = os.path.join(data_dir, f'{subject}_outdoor_baselines')
        src_dir = image_dir
        gt_dir = os.path.join(data_dir, f'{subject}_outdoor_eval')
        print(f'copying images from {src_dir} to {dst_dir}, gt from {gt_dir}')
        move_images_to_folder(src_dir, dst_dir, gt_dir, method_name=method_name)

        # step 2. crop images and store
        method_dir = os.path.join(dst_dir, method_name)
        create_cropped_images(method_dir, gt_dir)
    
    perceptual_results = evaluate_perfcap_kid_fid_subject(data_dir, [method_name], subject=subject)
    eval_results = evaluate_perfcap_subject(data_dir, [method_name], subject)
    eval_results[method_name].update(**perceptual_results[method_name])

    save_dir = os.path.join(output_dir, method_name)
    save_path = os.path.join(save_dir, f'{subject}.npy')
    os.makedirs(save_dir, exist_ok=True)
    np.save(save_path, eval_results, allow_pickle=True)
    print(f'Results saved to {save_path}')
    