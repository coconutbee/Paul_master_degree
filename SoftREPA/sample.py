import argparse
import csv
import numpy as np
import random, os
import glob
import re
import time
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.utils import save_image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as torch_transforms
import torch.nn as nn
import tqdm

from sampler import SD3EulerDC, SDXLEulerDC, SD1EulerDC
from dataset.datasets import get_target_dataset
import json

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def set_unreproducible_seed():
    # Draw seed from OS entropy so each inference is not reproducible.
    seed = int.from_bytes(os.urandom(8), byteorder='big') % (2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    return seed


def sanitize_filename(name: str, max_len: int = 120) -> str:
    # Keep filenames portable and safe across filesystems.
    cleaned = re.sub(r'[\\/:*?"<>|\n\r\t]', '_', name).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned[:max_len] if cleaned else 'empty_prompt'


def resolve_token_path(load_dir: str, stem: str) -> str:
    preferred = os.path.join(load_dir, f'{stem}.pth')
    if os.path.isfile(preferred):
        return preferred

    candidates = glob.glob(os.path.join(load_dir, f'{stem}_*.pth'))
    if not candidates:
        raise FileNotFoundError(f'Cannot find {stem}.pth or {stem}_*.pth under: {load_dir}')

    def extract_index(path: str) -> int:
        name = os.path.basename(path)
        match = re.search(rf'^{re.escape(stem)}_(\d+)\.pth$', name)
        return int(match.group(1)) if match else -1

    candidates.sort(key=extract_index)
    return candidates[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # sampling config
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=1024, choices=[256,512,768,1024])
    parser.add_argument('--NFE', type=int, default=28)
    parser.add_argument('--cfg_scale', type=float, default=1.0, help='0 for null prompt, 1 for only using conditional prompt')
    parser.add_argument('--batch_size', type=int, default=1)
    # path
    parser.add_argument('--load_dir', type=str, default=None, help="replace it with your checkpoint")
    parser.add_argument('--save_dir', type=str, default=None, help="default savedir is set to under load_dir")
    parser.add_argument('--datadir', type=str, default='', required=True, help='data path')
    # model config
    parser.add_argument('--model', type=str, default='sd3', choices=['sd3', 'sdxl', 'sd1.5'], help='Model to use')
    parser.add_argument('--use_dc', action='store_true', default=False)
    parser.add_argument('--use_dc_t', type=str, default=False, help='use t dependent')
    parser.add_argument('--n_dc_tokens', type=int, default=4)
    parser.add_argument('--n_dc_layers', type=int, default=5, help='sd3')
    parser.add_argument('--apply_dc', nargs='+', type=str, default=[True, True, False], help='sdxl, sd1.5')
    # one sample generation
    parser.add_argument('--prompt', type=str, default="")
    parser.add_argument('--save_name', type=str, default="image_sd3")
    # set generation
    parser.add_argument('--num', type=int, default=-1, help='number of sampling images. -1 for whole dataset')
    parser.add_argument('--dataset', type=str, nargs='+', default=None, choices=['coco'])
    
    args = parser.parse_args()
    # Keep this for compatibility, but actual inference now reseeds per image.
    set_seed(args.seed)

    args.apply_dc = [str2bool(x) for x in args.apply_dc]
    args.use_dc_t = str2bool(args.use_dc_t)

    interpolation = INTERPOLATIONS['bilinear']
    transform = get_transform(interpolation, 1024)

    # load model
    if args.model == 'sd3':
        sampler = SD3EulerDC(n_dc_tokens=args.n_dc_tokens, use_8bit=False, use_dc_t=args.use_dc_t, n_dc_layers=args.n_dc_layers)
    elif args.model == 'sdxl':
        sampler = SDXLEulerDC(n_dc_tokens=args.n_dc_tokens, use_8bit=False, use_dc_t=args.use_dc_t, apply_dc=args.apply_dc)
    elif args.model == 'sd1.5':
        sampler = SD1EulerDC(n_dc_tokens=args.n_dc_tokens, use_8bit=False, use_dc_t=args.use_dc_t, apply_dc=args.apply_dc)
    else:
        raise ValueError('args.model should be one of [sd3, sdxl, sd1.5]')
    
    # load tokens
    if args.load_dir is not None:
        dc_path = resolve_token_path(args.load_dir, 'dc_tokens')
        dc_tokens = torch.load(dc_path, map_location='cpu', weights_only=False)

        dc_t_tokens = None
        if args.use_dc_t:
            dc_t_path = resolve_token_path(args.load_dir, 'dc_t_tokens')
            dc_t_tokens = torch.load(dc_t_path, map_location='cpu', weights_only=False)

        # Align checkpoint tensors to model parameter device/dtype before initialize_dc.
        dc_tokens = dc_tokens.to(
            device=sampler.denoiser.dc_tokens.device,
            dtype=sampler.denoiser.dc_tokens.dtype,
        )
        if dc_t_tokens is not None:
            dc_t_tokens = dc_t_tokens.to(
                device=sampler.denoiser.dc_t_tokens.weight.device,
                dtype=sampler.denoiser.dc_t_tokens.weight.dtype,
            )

        # initialize_dc casts token dtype to match model dtype/device and avoids float/half mismatch.
        sampler.initialize_dc(dc_tokens, dc_t_tokens)
        print(f'Loaded dc tokens from: {dc_path}')
        if args.use_dc_t:
            print(f'Loaded dc_t tokens from: {dc_t_path}')
    
    # sample set
    if args.dataset is not None:
        # save dir
        config=f'{"-".join(args.dataset)}-cfg{args.cfg_scale}-dc{args.use_dc}-dct{args.use_dc_t}-nfe{args.NFE}'
        if args.save_dir is not None:
            args.load_dir = args.save_dir
        savedir = os.path.join(args.load_dir, config)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        train_datasets = []
        for ds in args.dataset:
            train_datasets.append(get_target_dataset(ds, args.datadir, train=False, transform=transform))

        train_dataset = ConcatDataset(train_datasets)
        num = args.num if args.num != -1 else len(train_dataset)
        train_dataset = Subset(train_dataset, list(range(num)))
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
        pbar = tqdm.tqdm(train_dataloader)
        i=0
        results = []
        for _, label in pbar:
            if os.path.exists(os.path.join(savedir, f'{i+args.batch_size:04d}.png')):
                i+=1
                continue
            current_seed = set_unreproducible_seed()
            img = sampler.sample(label, NFE=args.NFE, img_shape=(args.img_size, args.img_size), cfg_scale=args.cfg_scale, use_dc=args.use_dc, batch_size=len(label))
            for bi in range(img.shape[0]):
                imgname = f'{i:04d}.png'
                save_image(img[bi], os.path.join(savedir, imgname), normalize=True)
                results.append({"prompt": label[bi], "img_path": imgname})
                pbar.set_description(f'SD Sampling [{i}/{num}] seed={current_seed}')
                i+=1
        
        # save config
        if os.path.exists(os.path.join(args.load_dir, f"results-{config}.json")):
            with open(os.path.join(args.load_dir, f"results-{config}.json"), 'r', encoding='utf-8') as file:
                results_all = json.load(file)
                if isinstance(results_all, list):
                    results_all.extend(results)
                else:
                    results_all = [results_all] + results
        else:
            results_all = results
        with open(os.path.join(args.load_dir, f"results-{config}.json"), "w", encoding="utf-8") as file:
            json.dump(results_all, file, indent=4)  # `indent=4` makes the JSON more readable

    # sample image
    else:
        # save dir
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # If prompt file exists and no explicit prompt is provided, read prompts from CSV/JSONL.
        prompt_file_path = None
        prompt_file_type = None
        datadir_lower = args.datadir.lower()
        if datadir_lower.endswith('.csv'):
            prompt_file_path = args.datadir
            prompt_file_type = 'csv'
        elif datadir_lower.endswith('.jsonl'):
            prompt_file_path = args.datadir
            prompt_file_type = 'jsonl'
        else:
            csv_path = os.path.join(args.datadir, 'gt.csv')
            jsonl_path = os.path.join(args.datadir, 'gt.jsonl')
            if os.path.exists(csv_path):
                prompt_file_path = csv_path
                prompt_file_type = 'csv'
            elif os.path.exists(jsonl_path):
                prompt_file_path = jsonl_path
                prompt_file_type = 'jsonl'
            else:
                # Fallback: auto-detect the first prompt file in the directory.
                jsonl_candidates = sorted(glob.glob(os.path.join(args.datadir, '*.jsonl')))
                csv_candidates = sorted(glob.glob(os.path.join(args.datadir, '*.csv')))
                if jsonl_candidates:
                    prompt_file_path = jsonl_candidates[0]
                    prompt_file_type = 'jsonl'
                elif csv_candidates:
                    prompt_file_path = csv_candidates[0]
                    prompt_file_type = 'csv'

        if args.prompt == "" and prompt_file_path is not None and os.path.exists(prompt_file_path):
            prompts = []
            if prompt_file_type == 'csv':
                with open(prompt_file_path, 'r', encoding='utf-8-sig', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        _ = row.get('item', None)
                        prompt = row.get('prompt', '')
                        if not prompt:
                            # Fallback for alternate header names and unexpected casing/spaces.
                            for key, value in row.items():
                                key_norm = key.replace('\ufeff', '').strip().lower() if key is not None else ''
                                if key_norm == 'prompt':
                                    prompt = value
                                    break
                        if prompt:
                            prompts.append(prompt)
            elif prompt_file_type == 'jsonl':
                with open(prompt_file_path, 'r', encoding='utf-8-sig') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        prompt = row.get('prompt', '') if isinstance(row, dict) else ''
                        if prompt:
                            prompts.append(prompt)

            desc = 'CSV Sampling' if prompt_file_type == 'csv' else 'JSONL Sampling'
            pbar = tqdm.tqdm(prompts, total=len(prompts), desc=desc)
            total_time = 0.0
            generated = 0
            for prompt in pbar:
                current_seed = set_unreproducible_seed()
                t0 = time.time()
                img = sampler.sample([prompt], NFE=args.NFE, img_shape=(args.img_size, args.img_size), cfg_scale=args.cfg_scale, use_dc=args.use_dc, batch_size=1)
                img_name = f"{sanitize_filename(prompt)}.jpg"
                save_image(img, os.path.join(args.save_dir, img_name), normalize=True)
                dt = time.time() - t0
                total_time += dt
                generated += 1
                pbar.set_postfix(seed=current_seed, last_sec=f'{dt:.2f}', avg_sec=f'{(total_time / generated):.2f}')
        else:
            current_seed = set_unreproducible_seed()
            img = sampler.sample([args.prompt], NFE=args.NFE, img_shape=(args.img_size, args.img_size), cfg_scale=args.cfg_scale, use_dc=args.use_dc, batch_size=1)
            save_image(img, os.path.join(args.save_dir, f'{args.save_name}.png'), normalize=True)
            print(f'used_seed={current_seed}')
    