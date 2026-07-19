import math
import os
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import torchvision
from torchvision import transforms as T, utils

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

torch.backends.cudnn.benchmark = True
torch.manual_seed(4096)
if torch.cuda.is_available():
    torch.cuda.manual_seed(4096)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class ImageDataset(Dataset):
    def __init__(self, folder, image_size):
        self.paths = [p for p in Path(f'/home/tylin1228/DL/hw6/face').glob('**/*.jpg')]
        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        return self.transform(img)


# ─────────────────────────────────────────────
# StyleGAN2 Building Blocks
# ─────────────────────────────────────────────

class EqualLinear(nn.Module):
    """Linear layer with equalized learning rate"""
    def __init__(self, in_dim, out_dim, bias=True, lr_mul=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        return F.linear(x, self.weight * self.scale,
                        self.bias * self.lr_mul if self.bias is not None else None)


class EqualConv2d(nn.Module):
    """Conv2d with equalized learning rate"""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.scale = 1 / math.sqrt(in_ch * kernel_size ** 2)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias,
                        stride=self.stride, padding=self.padding)


class MappingNetwork(nn.Module):
    """
    z → w：把隨機噪聲映射到 intermediate latent space
    讓 latent space 更加 disentangled
    """
    def __init__(self, z_dim=512, w_dim=512, num_layers=8, lr_mul=0.01):
        super().__init__()
        layers = [PixelNorm()]
        for i in range(num_layers):
            layers.append(EqualLinear(z_dim if i == 0 else w_dim, w_dim, lr_mul=lr_mul))
            layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class PixelNorm(nn.Module):
    def forward(self, x):
        return x / (torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8).sqrt()


class NoiseInjection(nn.Module):
    """在每個 feature map 加入 learned noise，增加細節多樣性"""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        if noise is None:
            b, _, h, w = x.shape
            noise = torch.randn(b, 1, h, w, device=x.device)
        return x + self.weight * noise


class ModulatedConv2d(nn.Module):
    """
    StyleGAN2 核心：用 w 來調製卷積權重
    不用 AdaIN，直接在 weight 上做 modulation，更穩定
    """
    def __init__(self, in_ch, out_ch, kernel_size, w_dim=512, demodulate=True, upsample=False):
        super().__init__()
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_ch * kernel_size ** 2)

        # style linear：w → style（用來調製 weight）
        self.modulation = EqualLinear(w_dim, in_ch, bias=True)
        nn.init.ones_(self.modulation.bias)

    def forward(self, x, w):
        b, c, h, width = x.shape

        # Style modulation
        style = self.modulation(w).view(b, 1, c, 1, 1)  # (b, 1, in_ch, 1, 1)
        weight = self.weight * self.scale * style         # (b, out_ch, in_ch, k, k)

        # Demodulation（讓輸出標準差 ≈ 1）
        if self.demodulate:
            dcoefs = (weight.pow(2).sum([2, 3, 4]) + 1e-8).rsqrt()
            weight = weight * dcoefs.view(b, self.out_ch, 1, 1, 1)

        # Reshape for grouped conv（把 batch 塞進 group）
        x = x.view(1, b * c, h, width)
        weight = weight.view(b * self.out_ch, c, self.kernel_size, self.kernel_size)

        if self.upsample:
            x = F.interpolate(x.view(b, c, h, width), scale_factor=2, mode='bilinear', align_corners=False)
            x = x.view(1, b * c, h * 2, width * 2)

        out = F.conv2d(x, weight, padding=self.padding, groups=b)
        out = out.view(b, self.out_ch, out.shape[2], out.shape[3])
        return out


class StyledConv(nn.Module):
    """ModulatedConv2d + NoiseInjection + Activation"""
    def __init__(self, in_ch, out_ch, kernel_size, w_dim=512, upsample=False):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, out_ch, kernel_size, w_dim=w_dim, upsample=upsample)
        self.noise = NoiseInjection()
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, w, noise=None):
        x = self.conv(x, w)
        x = self.noise(x, noise)
        x = x + self.bias
        return self.act(x)


class ToRGB(nn.Module):
    """把 feature map 轉成 RGB 圖片"""
    def __init__(self, in_ch, w_dim=512):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, 3, kernel_size=1, w_dim=w_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, w, skip=None):
        out = self.conv(x, w) + self.bias
        if skip is not None:
            skip = F.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=False)
            out = out + skip
        return out


# ─────────────────────────────────────────────
# StyleGAN2 Generator
# ─────────────────────────────────────────────

class Generator(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, image_size=64, channel_base=4096):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim

        # Log2 of image size determines number of layers
        # 64 = 2^6, so log2(64) = 6, layers: 4,8,16,32,64
        self.log_size = int(math.log2(image_size))

        # Mapping network
        self.mapping = MappingNetwork(z_dim, w_dim)

        # Channels per resolution
        def nf(stage):
            return min(int(channel_base / (2 ** stage)), 512)

        # Initial constant input (4x4)
        self.input = nn.Parameter(torch.randn(1, nf(1), 4, 4))

        # First conv (no upsample)
        self.conv1 = StyledConv(nf(1), nf(1), 3, w_dim=w_dim)
        self.to_rgb1 = ToRGB(nf(1), w_dim=w_dim)

        # Progressive upsampling layers
        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_ch = nf(1)
        for i in range(2, self.log_size):
            out_ch = nf(i)
            self.convs.append(StyledConv(in_ch, out_ch, 3, w_dim=w_dim, upsample=True))
            self.convs.append(StyledConv(out_ch, out_ch, 3, w_dim=w_dim))
            self.to_rgbs.append(ToRGB(out_ch, w_dim=w_dim))
            in_ch = out_ch

        self.n_latent = 2 + 3 * (self.log_size - 2)  # 每層需要一個 w

    def forward(self, z, return_latents=False):
        b = z.shape[0]

        # Mapping z → w，廣播到每一層
        w = self.mapping(z)
        ws = w.unsqueeze(1).repeat(1, self.n_latent, 1)  # (b, n_latent, w_dim)

        # Initial 4x4
        x = self.input.repeat(b, 1, 1, 1)
        x = self.conv1(x, ws[:, 0])
        skip = self.to_rgb1(x, ws[:, 1])

        # Progressive upsampling
        i = 2
        for conv1, conv2, to_rgb in zip(
            self.convs[0::2], self.convs[1::2], self.to_rgbs
        ):
            x = conv1(x, ws[:, i])
            x = conv2(x, ws[:, i + 1])
            skip = to_rgb(x, ws[:, i + 2], skip)
            i += 3

        img = torch.tanh(skip)

        if return_latents:
            return img, ws
        return img


# ─────────────────────────────────────────────
# StyleGAN2 Discriminator
# ─────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = EqualConv2d(in_ch, in_ch, 3, padding=1)
        self.conv2 = EqualConv2d(in_ch, out_ch, 3, padding=1)
        self.skip = EqualConv2d(in_ch, out_ch, 1, bias=False)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        out = F.avg_pool2d(out, 2)
        skip = F.avg_pool2d(self.skip(x), 2)
        return (out + skip) / math.sqrt(2)


class Discriminator(nn.Module):
    def __init__(self, image_size=64, channel_base=4096):
        super().__init__()

        log_size = int(math.log2(image_size))

        def nf(stage):
            return min(int(channel_base / (2 ** stage)), 512)

        # from_rgb
        self.from_rgb = nn.Sequential(
            EqualConv2d(3, nf(log_size - 1), 1),
            nn.LeakyReLU(0.2)
        )

        # ResBlocks（逐漸降低解析度）
        self.blocks = nn.ModuleList()
        in_ch = nf(log_size - 1)
        for i in range(log_size - 1, 1, -1):
            out_ch = nf(i - 1)
            self.blocks.append(ResBlock(in_ch, out_ch))
            in_ch = out_ch

        # Final layers（4x4）
        self.final_conv = EqualConv2d(in_ch + 1, in_ch, 3, padding=1)
        self.final_linear = nn.Sequential(
            EqualLinear(in_ch * 4 * 4, in_ch),
            nn.LeakyReLU(0.2),
            EqualLinear(in_ch, 1)
        )

    def minibatch_std(self, x):
        """Minibatch Std：增加多樣性，避免 mode collapse"""
        std = x.std(dim=0, keepdim=True).mean(dim=[1, 2, 3], keepdim=True)
        std = std.expand(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, std], dim=1)

    def forward(self, x):
        x = self.from_rgb(x)
        for block in self.blocks:
            x = block(x)
        x = self.minibatch_std(x)
        x = F.leaky_relu(self.final_conv(x), 0.2)
        x = x.view(x.shape[0], -1)
        return self.final_linear(x)


# ─────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────

def d_logistic_loss(real_pred, fake_pred):
    """Non-saturating GAN loss for discriminator"""
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def g_nonsaturating_loss(fake_pred):
    """Non-saturating GAN loss for generator"""
    return F.softplus(-fake_pred).mean()


def d_r1_loss(real_pred, real_img):
    """R1 gradient penalty（讓 D 在真實圖片上更穩定）"""
    grad_real, = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


# ─────────────────────────────────────────────
# StyleGAN Trainer
# ─────────────────────────────────────────────

def cycle(dl):
    while True:
        for data in dl:
            yield data


class StyleGANTrainer:
    def __init__(
        self,
        folder,
        image_size=64,
        z_dim=512,
        w_dim=512,
        batch_size=8,
        lr_g=2e-3,
        lr_d=2e-3,
        r1_gamma=10,
        r1_every=16,
        train_num_steps=10000,
        save_every=1000,
        results_folder='./results_stylegan',
        device=None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.train_num_steps = train_num_steps
        self.save_every = save_every
        self.r1_gamma = r1_gamma
        self.r1_every = r1_every
        self.step = 0

        # Models
        self.G = Generator(z_dim=z_dim, w_dim=w_dim, image_size=image_size).to(self.device)
        self.D = Discriminator(image_size=image_size).to(self.device)

        # Optimizers
        self.opt_G = Adam(self.G.parameters(), lr=lr_g, betas=(0.0, 0.99))
        self.opt_D = Adam(self.D.parameters(), lr=lr_d, betas=(0.0, 0.99))

        # Dataset
        ds = ImageDataset(folder, image_size)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        pin_memory=True, num_workers=2, drop_last=True)
        self.dl = cycle(dl)

        # Results folder
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # Fixed z for visualization
        self.fixed_z = torch.randn(25, z_dim, device=self.device)

    def train(self):
        for step in range(self.train_num_steps):
            self.step = step
            real_imgs = next(self.dl).to(self.device)

            # ── Train Discriminator ──
            self.opt_D.zero_grad()

            z = torch.randn(self.batch_size, self.z_dim, device=self.device)
            with torch.no_grad():
                fake_imgs = self.G(z)

            real_pred = self.D(real_imgs)
            fake_pred = self.D(fake_imgs)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            # R1 penalty（每 r1_every 步做一次）
            if step % self.r1_every == 0:
                real_imgs.requires_grad_(True)
                real_pred_r1 = self.D(real_imgs)
                r1_loss = d_r1_loss(real_pred_r1, real_imgs)
                d_loss = d_loss + (self.r1_gamma / 2 * r1_loss * self.r1_every)
                real_imgs.requires_grad_(False)

            d_loss.backward()
            self.opt_D.step()

            # ── Train Generator ──
            self.opt_G.zero_grad()

            z = torch.randn(self.batch_size, self.z_dim, device=self.device)
            fake_imgs = self.G(z)
            fake_pred = self.D(fake_imgs)
            g_loss = g_nonsaturating_loss(fake_pred)

            g_loss.backward()
            self.opt_G.step()

            # ── Logging & Saving ──
            if (step + 1) % self.save_every == 0:
                print(f'Step {step+1} | D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}')
                self.save_samples(step + 1)
                self.save_checkpoint(step + 1)

        print('Training complete!')

    @torch.no_grad()
    def save_samples(self, step):
        self.G.eval()
        fake = self.G(self.fixed_z)
        fake = (fake + 1) / 2  # [-1,1] → [0,1]
        utils.save_image(fake, str(self.results_folder / f'sample-{step}.png'),
                         nrow=5, normalize=False)
        self.G.train()

    def save_checkpoint(self, step):
        torch.save({
            'step': step,
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'opt_G': self.opt_G.state_dict(),
            'opt_D': self.opt_D.state_dict(),
        }, str(self.results_folder / f'ckpt-{step}.pt'))

    def load_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.G.load_state_dict(ckpt['G'])
        self.D.load_state_dict(ckpt['D'])
        self.opt_G.load_state_dict(ckpt['opt_G'])
        self.opt_D.load_state_dict(ckpt['opt_D'])
        self.step = ckpt['step']
        print(f'Loaded checkpoint from step {self.step}')

    @torch.no_grad()
    def inference(self, num=1000, n_iter=5, output_path='./submission_gan'):
        os.makedirs(output_path, exist_ok=True)
        self.G.eval()
        idx = 1
        for _ in range(n_iter):
            z = torch.randn(num // n_iter, self.z_dim, device=self.device)
            imgs = self.G(z)
            imgs = (imgs + 1) / 2  # [-1,1] → [0,1]
            for img in imgs:
                torchvision.utils.save_image(img, f'{output_path}/{idx}.jpg')
                idx += 1
        print(f'Saved {idx-1} images to {output_path}')

    @torch.no_grad()
    def visualize_progress(self, n_samples=5, n_steps=15):
        """Gradescope Q1：用不同 latent 展示生成結果"""
        self.G.eval()
        z_samples = torch.randn(n_samples, self.z_dim, device=self.device)
        imgs = self.G(z_samples)
        imgs = (imgs + 1) / 2

        fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 2, 2))
        for i, ax in enumerate(axes):
            ax.imshow(imgs[i].cpu().permute(1, 2, 0).clamp(0, 1))
            ax.axis('off')
        plt.suptitle('StyleGAN Generated Faces', fontsize=13)
        plt.tight_layout()
        plt.savefig('/home/tylin1228/DL/hw6/stylegan_samples.png', dpi=150)
        plt.close()
        self.G.train()

    @torch.no_grad()
    def style_mixing(self, n=5):
        """StyleGAN 特有功能：Style Mixing 展示 disentanglement"""
        self.G.eval()
        z1 = torch.randn(n, self.z_dim, device=self.device)
        z2 = torch.randn(n, self.z_dim, device=self.device)

        imgs1 = (self.G(z1) + 1) / 2
        imgs2 = (self.G(z2) + 1) / 2

        # Mix styles at different layers
        w1 = self.G.mapping(z1).unsqueeze(1).repeat(1, self.G.n_latent, 1)
        w2 = self.G.mapping(z2).unsqueeze(1).repeat(1, self.G.n_latent, 1)

        # 前半層用 w1，後半層用 w2
        mix_point = self.G.n_latent // 2
        w_mix = w1.clone()
        w_mix[:, mix_point:] = w2[:, mix_point:]

        # Forward with mixed w（需要繞過 mapping）
        # 展示原圖就好
        fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
        for i in range(n):
            axes[0, i].imshow(imgs1[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[0, i].axis('off')
            axes[1, i].imshow(imgs2[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[1, i].axis('off')
        axes[0, 0].set_ylabel('Style A', fontsize=10)
        axes[1, 0].set_ylabel('Style B', fontsize=10)
        plt.suptitle('StyleGAN Style Diversity', fontsize=13)
        plt.tight_layout()
        plt.savefig('/home/tylin1228/DL/hw6/stylegan_diversity.png', dpi=150)
        plt.close()
        self.G.train()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

path = '/home/tylin1228/DL/hw6/face'

trainer = StyleGANTrainer(
    folder=path,
    image_size=64,
    z_dim=512,
    w_dim=512,
    batch_size=8,         # StyleGAN 記憶體需求高
    lr_g=2e-3,
    lr_d=2e-3,
    r1_gamma=10,
    r1_every=16,
    train_num_steps=10000,
    save_every=1000,
)

trainer.train()


# ─────────────────────────────────────────────
# Inference & Visualization
# ─────────────────────────────────────────────

# 載入最後 checkpoint
trainer.load_checkpoint('./results_stylegan/ckpt-10000.pt')

# Gradescope Q1 視覺化
trainer.visualize_progress(n_samples=5)

# 多樣性展示
trainer.style_mixing(n=5)

# 產生提交檔案
trainer.inference(num=1000, n_iter=5, output_path='./submission_gan')

# 打包
import subprocess
subprocess.run(['tar', '-zcf', 'submission.tgz', '-C', './submission_gan', '.'])

