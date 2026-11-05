from denoising_diffusion_pytorch import Unet, GaussianDiffusion

def build_model(config):

    model = Unet(
        dim=config["model"]["dim"],
        dim_mults=config["model"]["dim_mults"],
        flash_attn=config["model"]["flash_attn"]
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=config["diffusion"]["image_size"],
        timesteps=config["diffusion"]["timesteps"]
    )

    return diffusion
