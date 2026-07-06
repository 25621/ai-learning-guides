"""img2img and inpainting built by hand from Stable Diffusion's raw parts.

No StableDiffusionImg2ImgPipeline, no inpainting checkpoint — just the VAE,
the U-Net, the text encoder, and a scheduler, wired together the way the LDM
formulation says. The point of the project: both features are ~15-line
modifications of the vanilla text-to-image denoising loop.

    img2img:    don't start from pure noise at t = T. Encode the init image,
                noise it to an intermediate level ("strength"), and denoise
                from there. strength=1 is txt2img; strength=0 returns the input.

    inpainting: run the normal loop, but after every step overwrite the
                KNOWN region's latents with the original image, re-noised to
                the current level. Only the masked hole is ever generated.
"""

import time
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

HERE = Path(__file__).resolve().parent
MODEL_ID = "segmind/tiny-sd"
SIZE = 384


class RawSD:
    """The four SD components, plus encode/decode/embed helpers."""

    def __init__(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32,
            safety_checker=None, requires_safety_checker=False,
        )
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.vae, self.unet = pipe.vae, pipe.unet
        self.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        self.scale = pipe.vae.config.scaling_factor  # SD's 0.18215

    @torch.no_grad()
    def embed(self, prompt: str):
        """(uncond, cond) text embeddings for CFG."""
        tok = self.pipe.tokenizer
        enc = self.pipe.text_encoder
        ids = tok([""], padding="max_length", max_length=tok.model_max_length,
                  return_tensors="pt").input_ids
        uncond = enc(ids)[0]
        ids = tok([prompt], padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids
        cond = enc(ids)[0]
        return uncond, cond

    @torch.no_grad()
    def encode_image(self, pil_image):
        import numpy as np

        x = torch.from_numpy(np.array(pil_image)).float() / 127.5 - 1.0
        x = x.permute(2, 0, 1)[None]
        return self.vae.encode(x).latent_dist.mean * self.scale

    @torch.no_grad()
    def decode(self, z):
        x = self.vae.decode(z / self.scale).sample.clamp(-1, 1)
        x = ((x[0].permute(1, 2, 0) + 1) * 127.5).byte().numpy()
        from PIL import Image

        return Image.fromarray(x)

    @torch.no_grad()
    def cfg_eps(self, z, t, uncond, cond, guidance=7.5):
        eps_u = self.unet(z, t, encoder_hidden_states=uncond).sample
        eps_c = self.unet(z, t, encoder_hidden_states=cond).sample
        return eps_u + guidance * (eps_c - eps_u)

    @torch.no_grad()
    def txt2img(self, prompt: str, steps: int = 15, guidance: float = 7.5,
                seed: int = 0, size: int = SIZE):
        """The vanilla loop both features below are modifications of."""
        uncond, cond = self.embed(prompt)
        self.scheduler.set_timesteps(steps)
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(1, 4, size // 8, size // 8, generator=g)
        z = z * self.scheduler.init_noise_sigma
        for t in self.scheduler.timesteps:
            eps = self.cfg_eps(z, t, uncond, cond, guidance)
            z = self.scheduler.step(eps, t, z).prev_sample
        return self.decode(z)

    @torch.no_grad()
    def img2img(self, init_image, prompt: str, strength: float = 0.6,
                steps: int = 15, guidance: float = 7.5, seed: int = 0):
        """The vanilla loop with a late entry point."""
        uncond, cond = self.embed(prompt)
        self.scheduler.set_timesteps(steps)
        timesteps = self.scheduler.timesteps
        start = min(int(steps * strength), steps)  # how much of the loop to run
        timesteps = timesteps[steps - start:]

        z0 = self.encode_image(init_image)
        g = torch.Generator().manual_seed(seed)
        noise = torch.randn(z0.shape, generator=g)
        # Jump the init image straight to the entry noise level.
        z = self.scheduler.add_noise(z0, noise, timesteps[:1]) if start > 0 else z0

        for t in timesteps:
            eps = self.cfg_eps(z, t, uncond, cond, guidance)
            z = self.scheduler.step(eps, t, z).prev_sample
        return self.decode(z)

    @torch.no_grad()
    def inpaint(self, init_image, mask_latent, prompt: str, steps: int = 15,
                guidance: float = 7.5, seed: int = 0, z_full_start=None):
        """Blended latent inpainting: mask_latent is 1 where NEW content goes.

        `z_full_start` optionally seeds the unknown region (used by
        outpainting to start the border from reflected context instead of
        pure noise).
        """
        uncond, cond = self.embed(prompt)
        self.scheduler.set_timesteps(steps)
        z_known = self.encode_image(init_image) if init_image is not None else None
        if z_known is None:
            z_known = z_full_start
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(z_known.shape, generator=g) * self.scheduler.init_noise_sigma

        for i, t in enumerate(self.scheduler.timesteps):
            eps = self.cfg_eps(z, t, uncond, cond, guidance)
            z = self.scheduler.step(eps, t, z).prev_sample
            # Re-impose the known region at this step's noise level.
            if i + 1 < len(self.scheduler.timesteps):
                t_next = self.scheduler.timesteps[i + 1]
                noise = torch.randn(z.shape, generator=g)
                z_known_t = self.scheduler.add_noise(z_known, noise, t_next[None])
            else:
                z_known_t = z_known  # final step: paste the clean latents
            z = mask_latent * z + (1 - mask_latent) * z_known_t
        return self.decode(z)


def timed(fn, *a, **kw):
    t0 = time.time()
    out = fn(*a, **kw)
    return out, time.time() - t0
