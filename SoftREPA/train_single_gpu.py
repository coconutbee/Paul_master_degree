import argparse
import numpy as np
import os
import glob
import re
import yaml, json
import os.path as osp
import pandas as pd
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from torch.utils.data import ConcatDataset
from dataset.datasets import get_target_dataset

from sampler import SD3EulerDC, SDXLEulerDC, SD1EulerDC
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode

from util import set_seed, save_on_master

import ImageReward as RM
from eval_utils import PickScore, HPSv2

try:
    import wandb
except ImportError:
    wandb = None

device = "cuda" if torch.cuda.is_available() else "cpu"

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _extract_step_from_filename(path):
    match = re.search(r'dc_tokens_(\d+)\.pth$', osp.basename(path))
    if match is None:
        return -1
    return int(match.group(1))


def _find_latest_dc_path(run_dir):
    token_paths = glob.glob(osp.join(run_dir, 'dc_tokens_*.pth'))
    if not token_paths:
        return None
    token_paths.sort(key=_extract_step_from_filename)
    return token_paths[-1]


def _prompt_to_safe_filename(prompt, max_len=120):
    text = str(prompt).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9._ -]+', '_', text)
    text = text.replace(' ', '_')
    text = re.sub(r'_+', '_', text).strip('._-')
    if not text:
        text = 'empty_prompt'
    return text[:max_len].rstrip('._-') or 'empty_prompt'
        

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


def eval(args, model, target_dataset, eval_run_folder, **sample_cfg):
    pbar_eval = tqdm.tqdm(range(args.num_eval))
    eval_results = []
    used_names = set()
    for vi in pbar_eval:
        _, label = target_dataset[vi]
        with autocast(enabled=args.dtype == 'float16'):
            img = model.sampler.sample(label, null_prompt_emb=model.null_embs, **sample_cfg)

        if args.val_filename_use_prompt:
            base_name = _prompt_to_safe_filename(label)
            file_name = f'{base_name}.png'
            if file_name in used_names:
                file_name = f'{base_name}_{vi:04d}.png'
        else:
            file_name = f'{vi:04d}.png'
        used_names.add(file_name)

        save_image(img, osp.join(eval_run_folder, file_name), normalize=True)
        eval_results.append({"prompt": label, "img_path": file_name})
        pbar_eval.set_description(f'SD Evaluation Sampling [{vi}/{args.num_eval}]')

    benchmark_types = args.benchmark.split(",")
    benchmark_types = [x.strip() for x in benchmark_types]
    benchmark_results = {}
    for benchmark_type in benchmark_types:
        print('Benchmark Type: ', benchmark_type)
        eval_model = None
        reward_list = []
        if benchmark_type == "ImageReward-v1.0":
            eval_model = RM.load(name=benchmark_type, device="cuda")
        elif benchmark_type == "PickScore":
            eval_model = PickScore(device="cuda")
        elif benchmark_type == "HPS":
            eval_model = HPSv2()
        elif benchmark_type == 'CLIP':
            eval_model = RM.load_score(
                name=benchmark_type, device="cuda"
            )

        with torch.no_grad():
            for vi in range(args.num_eval):
                prompt = eval_results[vi]["prompt"]
                img_path = os.path.join(eval_run_folder, eval_results[vi]["img_path"])

                if benchmark_type in ["ImageReward-v1.0", "PickScore", "HPS"]:
                    rewards = eval_model.score(prompt, [img_path])
                else:
                    _, rewards = eval_model.inference_rank(prompt, [img_path])
                
                if isinstance(rewards, list):
                    rewards = float(rewards[0])

                reward_list.append(rewards)
        reward_list = np.array(reward_list)
        benchmark_results[benchmark_type] = reward_list.mean()
    return benchmark_results, eval_results


class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.07, scale=4.0, device='cuda', dweight=0):
        super().__init__()
        self.device = device
        self.temp = torch.nn.Parameter(torch.tensor(temp).to(self.device))
        self.scale = torch.nn.Parameter(torch.tensor(scale).to(self.device))
        self.dweight = dweight

    def get_mask(self, shape=None): # label: [b,], shape: (b, n_p)
        mask = torch.zeros(shape, device=self.device)
        n_b, n_p = shape
        index = torch.arange(n_b, device=self.device)
        mask[index, index] = 1
        return mask # (b, n_p)
        
    def forward(self, errors):
        # compute mask
        masks = self.get_mask(shape=errors.shape) # (b, n_p)        
        # compute logits
        logits = self.scale * torch.exp(-errors/self.temp)
        # compute loss
        loss = F.cross_entropy(logits, masks) 
        loss += self.dweight * errors[list(range(masks.shape[0])), list(range(masks.shape[0]))].mean()
        return loss


class SoftREPA(nn.Module):
    def __init__(self, sampler, device='cuda', dtype='float16'):
        super().__init__()
        self.sampler = sampler
        self.device = device
        self.dtype = dtype
        self.null_embs = self.sampler.encode_prompt([""])

    def forward(self, image, label, t, use_dc=True):
        with torch.no_grad():
            # compute image latent
            img_input = image.to(device)
            n_b,_,h,w = img_input.shape
            img_shape = (h,w)
            if self.dtype == 'float16':
                img_input = img_input.half()
            latent = self.sampler.encode(img_input)

            # compute prompt embeddings
            prompt_embs = self.sampler.encode_prompt(label)

            n_p, n_tkn, n_dim = prompt_embs[0].shape[-3:]
        
        # batch for contrastive learning (b, c, dim, dim) -> (n_p*b, c, dim, dim)
        batch_latent = torch.cat([latent]*n_p, 0)
        # batch for contrastive learning (n_p, n_tkn, n_dim) -> (n_p*b, n_tkn, n_dim)
        batch_pidxs = torch.arange(n_p, device=self.device).unsqueeze(0).repeat(n_b,1).transpose(0,1).contiguous().reshape(-1)

        # set noise and timestep
        self.sampler.set_noise(img_shape=img_shape, batch_size=1)
        batch_nidxs = torch.zeros(n_p*n_b, device=self.device).long().contiguous()
        v, pred_v = self.sampler.error(batch_latent, batch_nidxs, batch_pidxs, prompt_embs, t, use_dc=use_dc)
        error = F.mse_loss(v, pred_v, reduction='none').mean(dim=(1, 2, 3))

        return error.reshape(n_p, n_b).transpose(0,1) #(b, n_p)


def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--dataset', nargs='+', type=str, default=['deepfashion'], choices=['coco', 'deepfashion'], help='Dataset to use')
    parser.add_argument('--target_dataset', type=str, default='deepfashion', choices=['coco', 'deepfashion'], help='Dataset to use')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers to split the dataset across')

    # run args
    parser.add_argument('--model', type=str, default='sd3', choices=['sd3', 'sdxl', 'sd1.5'], help='Model to use')
    parser.add_argument('--n_dc_tokens', type=int, default=4, help='the number of learnable dc tokens')
    parser.add_argument('--n_dc_layers', type=int, default=5, help='the number of layers to append dc_tokens (sd3)')
    parser.add_argument('--apply_dc', nargs='+', type=str, default=[True, False, False], help='down/mid/up layers of unet (sd1.5, sdxl)')
    parser.add_argument('--use_dc_t', action='store_true', default=False, help='t dependent tokens')
    parser.add_argument('--max_t', type=int, default=1000)
    parser.add_argument('--min_t', type=int, default=0)
    parser.add_argument('--dweight', type=float, default=0, help='weight of diffusion score matching loss')

    # training args
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3, help='train learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512, 768, 1024), help='training image size')
    parser.add_argument('--batch_size', '-b', type=int, default=4)
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                        help='Model data type to use')
    parser.add_argument('--device', type=str, default='cuda', choices=('cuda', 'cpu'))

    # save/eval args
    parser.add_argument('--logdir', type=str, default='./data', help='path for save checkpoint')
    parser.add_argument('--datadir', type=str, default='', required=True, help='data path')
    parser.add_argument('--num_iter', type=int, default=2500, help='number of iterations before validation')
    parser.add_argument('--num_eval', type=int, default=50, help='number of generating images during validation')
    parser.add_argument('--benchmark', default="ImageReward-v1.0, CLIP, PickScore", type=str,
                        help="ImageReward-v1.0, Aesthetic, BLIP or CLIP, PickScore, HPS splitted with comma(,) if there are multiple benchmarks.")
    parser.add_argument('--note', type=str, default=None, help='note for saving path')
    parser.add_argument('--resume', action='store_true', default=False, help='resume from previous checkpoint in run folder or --resume_dir')
    parser.add_argument('--resume_dir', type=str, default=None, help='checkpoint folder to resume from (defaults to current run folder)')
    parser.add_argument('--resume_state', type=str, default=None, help='path to state checkpoint (.pth). default: <resume_dir>/run_state_latest.pth')
    parser.add_argument('--resume_dc_path', type=str, default=None, help='path to dc_tokens*.pth to initialize resumed tokens')
    parser.add_argument('--resume_dc_t_path', type=str, default=None, help='path to dc_t_tokens*.pth for --use_dc_t')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='log training and validation to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='SoftREPA', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity/team name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=('online', 'offline', 'disabled'), help='wandb mode')
    parser.add_argument('--wandb_log_images', type=int, default=8, help='number of validation images to upload each eval')
    parser.add_argument('--val_filename_use_prompt', type=str, default='True', help='whether to use prompt text as validation image filename')

    # multi gpus
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--separate_gpus', action='store_true', default=False, help='Use separate GPUs for each model')
    parser.add_argument('--use_8bit', action='store_true', default=False, help='Use 8bit quantization for T5 and transformer.')
    
    args = parser.parse_args()
    args.apply_dc = [str2bool(x) for x in args.apply_dc]
    args.val_filename_use_prompt = str2bool(args.val_filename_use_prompt)
    use_separate_mode = args.separate_gpus and torch.cuda.device_count() > 1
    if args.separate_gpus and not use_separate_mode:
        print('[WARN] --separate_gpus is set but fewer than 2 GPUs are visible. Falling back to single-GPU mode.')

    # setup
    set_seed(42)
    # if torch.cuda.device_count()>1 and not args.separate_gpus:
    #     init_distributed_mode(args)

    # make run output folder
    name = f"{args.model}"
    if args.img_size != 512:
        name += f'_{args.img_size}'
    name += f'_np{args.n_dc_tokens}'
    name += f'_nl{args.n_dc_layers}'
    name += f'_usedct{args.use_dc_t}'
    if args.note != None:
        name += f'_{args.note}'
    run_folder = osp.join(args.logdir, "-".join(args.dataset), name)
    os.makedirs(run_folder, exist_ok=True)
    print(f'Run folder: {run_folder}')

    resume_dir = args.resume_dir if args.resume_dir is not None else run_folder
    run_state_path = args.resume_state if args.resume_state is not None else osp.join(resume_dir, 'run_state_latest.pth')

    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            raise ImportError('wandb is not installed. Install it with: pip install wandb')
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
            dir=run_folder,
            mode=args.wandb_mode,
        )
        print(f'[WandB] Initialized run: {wandb_run.name}')

    # save arguments to a YAML file
    with open(os.path.join(run_folder, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f)
    print('Arguments saved to config.yaml')
    
    # set up dataset for train
    interpolation = INTERPOLATIONS[args.interpolation]
    transform = get_transform(interpolation, args.img_size)

    datasets =[]
    for ds in args.dataset:
        train_dataset = get_target_dataset(ds, args.datadir, train=True, transform=transform)
        datasets.append(train_dataset)
    train_dataset = ConcatDataset(datasets)

    dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                num_workers=args.n_workers,
                drop_last=True,
                pin_memory=True,
                shuffle=True)
    
    # set up dataset for eval
    target_dataset = get_target_dataset(args.target_dataset, args.datadir, train=False, transform=transform)
    target_dataloader = torch.utils.data.DataLoader(target_dataset, batch_size=1, shuffle=False)

    # load pretrained models
    if args.model == 'sd3':
        sampler = SD3EulerDC(n_dc_tokens=args.n_dc_tokens, n_dc_layers=args.n_dc_layers, use_dc_t=args.use_dc_t, use_8bit=args.use_8bit)
        sample_cfg = {'NFE':28, 'img_shape':(1024,1024), 'cfg_scale':4, 'use_dc':True}
        if use_separate_mode:
            sampler.text_enc_1.to("cuda:0")
            sampler.text_enc_2.to("cuda:0")
            sampler.vae.to("cuda:0")
            sampler.denoiser.to("cuda:1")
        model_device = sampler.denoiser.device

    elif args.model == 'sdxl':
        sampler = SDXLEulerDC(n_dc_tokens=args.n_dc_tokens, use_8bit=args.use_8bit, use_dc_t=args.use_dc_t, apply_dc=args.apply_dc)
        sample_cfg = {'NFE':30, 'img_shape':(1024,1024), 'cfg_scale':7.0, 'use_dc':True} 
        if use_separate_mode:
            sampler.text_enc.to("cuda:0")
            sampler.text_enc_2.to("cuda:0")
            sampler.vae.to("cuda:0")
            sampler.denoiser.to("cuda:1")
        model_device = sampler.denoiser.device

    elif args.model == 'sd1.5':
        sampler = SD1EulerDC(n_dc_tokens=args.n_dc_tokens, use_8bit=args.use_8bit, use_dc_t=args.use_dc_t, apply_dc=args.apply_dc)
        sample_cfg = {'NFE':30, 'img_shape':(512,512), 'cfg_scale':7.0, 'use_dc':True} 
        if use_separate_mode:
            sampler.text_enc.to("cuda:0")
            sampler.vae.to("cuda:0")
            sampler.denoiser.to("cuda:1")
        model_device = sampler.denoiser.device

    resume_state = None
    if args.resume:
        if osp.isfile(run_state_path):
            resume_state = torch.load(run_state_path, map_location='cpu', weights_only=False)
            print(f'[RESUME] Loaded run state: {run_state_path}')
        else:
            print(f'[RESUME] Run state not found: {run_state_path}. Resume will continue with token-only loading if available.')

        dc_path = args.resume_dc_path
        if dc_path is None and isinstance(resume_state, dict):
            dc_path = resume_state.get('last_dc_path')
        if dc_path is None:
            dc_path = _find_latest_dc_path(resume_dir)
        if dc_path is None or not osp.isfile(dc_path):
            raise FileNotFoundError(f'[RESUME] Could not find dc token checkpoint. Provide --resume_dc_path or make sure {resume_dir} has dc_tokens_*.pth')

        dc_tokens = torch.load(dc_path, map_location='cpu', weights_only=False)

        dc_t_path = args.resume_dc_t_path
        if dc_t_path is None and isinstance(resume_state, dict):
            dc_t_path = resume_state.get('last_dc_t_path')
        dc_t_tokens = None
        if args.use_dc_t:
            if dc_t_path is not None and osp.isfile(dc_t_path):
                dc_t_tokens = torch.load(dc_t_path, map_location='cpu', weights_only=False)
            else:
                print('[RESUME] --use_dc_t is enabled but dc_t checkpoint was not found. Continuing without dc_t resume.')

        with torch.no_grad():
            sampler.initialize_dc(dc_tokens, dc_t_tokens)
        print(f'[RESUME] Loaded dc tokens from: {dc_path}')
        if args.use_dc_t and dc_t_tokens is not None:
            print(f'[RESUME] Loaded dc_t tokens from: {dc_t_path}')
    elif args.model == 'sd1.5':
        with torch.no_grad():
            null_embs = sampler.encode_prompt([""])[0][0, :1].expand(args.n_dc_tokens, -1)
            sampler.initialize_dc(null_embs)

    # set up for contrastive learning
    model = SoftREPA(sampler, device=args.device, dtype=args.dtype)
    scaler = GradScaler() if args.dtype == 'float16' else None
    model = model.to(args.device)
    loss_criterion = ContrastiveLoss(device=model_device, dweight=args.dweight)

    # set requires grad
    update_param_set = []
    for name, param in model.sampler.denoiser.named_parameters():
        if 'dc' in name:
            param.requires_grad = True
            print(f'param: {name} requires grad [True]')
            update_param_set.append({'params':param})
        else:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(update_param_set, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5) # T_0=20, T_mult=2

    # multi gpu
    if use_separate_mode:
        print('Using separate-gpu mode with {} GPUs'.format(torch.cuda.device_count()))
    else:
        print('Using single-gpu mode')
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.sampler.denoiser.parameters() if p.requires_grad)}")

    model.sampler.denoiser.train()

    save_dict = {'epoch':[], 'loss':[], 'lr':[]}
    benchmark_types = args.benchmark.split(',')
    benchmark_types = [x.strip() for x in benchmark_types]
    for bch in benchmark_types:
        save_dict[bch] = []
    # train
    best_acc = 0.0
    iteration = 0
    global_loss = 0.0

    if args.resume and isinstance(resume_state, dict):
        # Restore optimizer/scheduler/scaler and bookkeeping for seamless continuation.
        if 'optimizer' in resume_state:
            optimizer.load_state_dict(resume_state['optimizer'])
        if 'lr_scheduler' in resume_state:
            lr_scheduler.load_state_dict(resume_state['lr_scheduler'])
        if scaler is not None and 'scaler' in resume_state and resume_state['scaler'] is not None:
            scaler.load_state_dict(resume_state['scaler'])
        if 'save_dict' in resume_state and isinstance(resume_state['save_dict'], dict):
            save_dict = resume_state['save_dict']
        iteration = int(resume_state.get('iteration', 0))
        best_acc = float(resume_state.get('best_acc', 0.0))
        global_loss = float(resume_state.get('global_loss', 0.0))
        print(f'[RESUME] iteration={iteration}, best_acc={best_acc:.4f}')

        if args.use_wandb and wandb_run is not None:
            wandb_run.summary['resume_iteration'] = iteration
            wandb_run.summary['resume_best_acc'] = best_acc

    for ep in range(args.epochs):
        pbar = tqdm.tqdm(dataloader)
        for i, (image, label) in enumerate(pbar):
            iteration += 1
            optimizer.zero_grad()
            image = image.to(args.device)
            with autocast(enabled=args.dtype == 'float16'):
                t = torch.randint(args.min_t, args.max_t, (1,), device=args.device)
                errors = model(image, label, t.long())
                loss = loss_criterion(errors)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            global_loss += loss.item()
            pbar.set_description(f'Loss: {loss.item():.4f}')

            if args.use_wandb and wandb_run is not None:
                wandb.log({'train/loss_step': loss.item()}, step=iteration)

            # validation
            if iteration % args.num_iter == 0: 
                it_ep = iteration // args.num_iter
                # save model
                global_loss /= args.num_iter
                print(f'Epoch {ep} Iteration {iteration}: Loss: {global_loss:.4f}')
                saved_dc_path = osp.join(run_folder, f'dc_tokens_{it_ep}.pth')
                save_on_master(sampler.denoiser.dc_tokens, saved_dc_path)
                saved_dc_t_path = None
                if args.use_dc_t:
                    saved_dc_t_path = osp.join(run_folder, f'dc_t_tokens_{it_ep}.pth')
                    save_on_master(sampler.denoiser.dc_t_tokens, saved_dc_t_path)
                
                # evaluate
                eval_run_folder = osp.join(run_folder, f'val_samples_{it_ep}')
                os.makedirs(eval_run_folder, exist_ok=True)

                benchmark_results, eval_results = eval(args, model, target_dataset, eval_run_folder, **sample_cfg)

                for bch,result in benchmark_results.items():
                    save_dict[bch].append(result) 
                
                save_dict['epoch'].append(it_ep)
                save_dict['loss'].append(global_loss)
                current_lr = optimizer.param_groups[0]['lr']
                save_dict['lr'].append(current_lr)
                df = pd.DataFrame(save_dict)
                df.to_csv(os.path.join(run_folder, 'run.csv'), index=False)

                # save best model
                acc=save_dict['CLIP'][-1]
                if acc > best_acc:
                    print('Save best checkpoint!')
                    best_acc = acc
                    save_on_master(sampler.denoiser.dc_tokens, osp.join(run_folder, 'dc_tokens_best.pth'))
                    if args.use_dc_t:
                        save_on_master(sampler.denoiser.dc_t_tokens, osp.join(run_folder, f'dc_t_tokens_best.pth'))

                # reset loss
                global_loss = 0.0
                lr_scheduler.step()

                if args.use_wandb and wandb_run is not None:
                    metric_log = {
                        'train/loss_eval_window': save_dict['loss'][-1],
                        'train/lr': current_lr,
                        'train/best_clip': best_acc,
                    }
                    for bch, result in benchmark_results.items():
                        metric_log[f'val/{bch}'] = result

                    n_img = max(0, min(args.wandb_log_images, len(eval_results)))
                    if n_img > 0:
                        metric_log['val/images'] = [
                            wandb.Image(
                                osp.join(eval_run_folder, eval_results[idx]['img_path']),
                                caption=str(eval_results[idx]['prompt'])
                            )
                            for idx in range(n_img)
                        ]
                    wandb.log(metric_log, step=iteration)

                state_payload = {
                    'iteration': iteration,
                    'best_acc': best_acc,
                    'global_loss': global_loss,
                    'save_dict': save_dict,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'scaler': scaler.state_dict() if scaler is not None else None,
                    'last_dc_path': saved_dc_path,
                    'last_dc_t_path': saved_dc_t_path,
                    'args': vars(args),
                }
                torch.save(state_payload, osp.join(run_folder, 'run_state_latest.pth'))

    print(f'Best accuracy: {best_acc:.2f}')
    print(f'Training complete. Saving model to {run_folder}')

    if args.use_wandb and wandb_run is not None:
        wandb_run.summary['final_best_acc'] = best_acc
        wandb.finish()

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
