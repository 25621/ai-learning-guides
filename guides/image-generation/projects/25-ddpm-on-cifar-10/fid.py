"""FID between generated samples and real CIFAR-10 images.

FID = Frechet distance between two Gaussians fitted to InceptionV3 pool
features of real and generated images:

    FID = ||mu_r - mu_g||^2 + Tr(C_r + C_g - 2 (C_r C_g)^{1/2})

The matrix square root is computed with a symmetric-eigendecomposition
identity, so no scipy dependency. Note the estimate is biased upward at small
sample counts — publishable numbers use 50k samples; be consistent about n
when comparing runs.

Run:
    python fid.py --ckpt checkpoints/cifar_smoke.pt --n 256
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "24-ddpm-on-mnist"))
from diffusion import GaussianDiffusion  # noqa: E402
from unet import UNet  # noqa: E402

from train_cifar import cifar_dataset  # noqa: E402


@torch.no_grad()
def inception_features(images: torch.Tensor, extractor, batch_size: int = 32) -> torch.Tensor:
    """images in [-1, 1], shape (N, 3, H, W) -> (N, 2048) pool features."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    feats = []
    for i in range(0, images.size(0), batch_size):
        x = (images[i : i + batch_size] + 1) / 2  # -> [0, 1]
        x = F.interpolate(x, size=299, mode="bilinear", align_corners=False)
        x = (x - mean) / std
        feats.append(extractor(x)["feat"].flatten(1))
    return torch.cat(feats)


def gaussian_stats(feats: torch.Tensor):
    mu = feats.mean(dim=0)
    centered = feats - mu
    cov = centered.T @ centered / (feats.size(0) - 1)
    return mu, cov


def sqrtm_psd(mat: torch.Tensor) -> torch.Tensor:
    """Square root of a (near-)PSD symmetric matrix via eigendecomposition."""
    vals, vecs = torch.linalg.eigh((mat + mat.T) / 2)
    return vecs @ torch.diag(vals.clamp(min=0).sqrt()) @ vecs.T

def frechet_distance(mu1, cov1, mu2, cov2) -> float:
    """Tr((C1 C2)^{1/2}) computed as Tr((C1^{1/2} C2 C1^{1/2})^{1/2}), which
    keeps everything symmetric PSD."""
    s1 = sqrtm_psd(cov1)
    covmean_tr = sqrtm_psd(s1 @ cov2 @ s1).trace()
    return ((mu1 - mu2).square().sum() + cov1.trace() + cov2.trace() - 2 * covmean_tr).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/cifar_smoke.pt")
    ap.add_argument("--n", type=int, default=256, help="generated sample count")
    ap.add_argument("--n-real", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=256, help="sampling batch")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # 1) generate
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=True)
    model = UNet(**ckpt["config"]).to(args.device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    diffusion = GaussianDiffusion(T=ckpt["T"], schedule=ckpt["schedule"], device=args.device)
    fakes = []
    for i in range(0, args.n, args.batch_size):
        b = min(args.batch_size, args.n - i)
        x0, _ = diffusion.p_sample_loop(model, (b, 3, 32, 32), device=args.device)
        fakes.append(x0.clamp(-1, 1))
        print(f"generated {i + b}/{args.n}", flush=True)
    fakes = torch.cat(fakes)

    # 2) real reference batch
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    ds = cifar_dataset(args.data_dir, train=True, transform=tf)
    loader = DataLoader(ds, batch_size=args.n_real, shuffle=True)
    reals, _ = next(iter(loader))

    # 3) features + Frechet distance
    inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    inception.eval()
    extractor = create_feature_extractor(inception, return_nodes={"avgpool": "feat"})
    mu_g, cov_g = gaussian_stats(inception_features(fakes, extractor))
    mu_r, cov_r = gaussian_stats(inception_features(reals, extractor))
    fid = frechet_distance(mu_r, cov_r, mu_g, cov_g)
    print(f"FID ({args.n} generated vs {args.n_real} real): {fid:.1f}")


if __name__ == "__main__":
    main()
