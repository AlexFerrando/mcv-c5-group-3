import consts
from diffusers import DiffusionPipeline

def read_prompts(user: str, filename: str) -> list[str]:
    """
    Read prompts from a file.
    Args:
        user (str): The user for path management.
        filename (str): The name of the file containing prompts.
    Returns:
        list: A list of prompts.
    """
    prompts_path = getattr(consts, f'PROMPTS_PATH_{user.upper()}')
    with open(f"{prompts_path}/{filename}", 'r') as file:
        prompts = file.readlines()
    return [prompt.strip() for prompt in prompts]


import torch
import os
from diffusers import DDIMScheduler, DDPMScheduler, DiffusionPipeline
from PIL import Image

def run_ddim_vs_ddpm_experiment(pipe: DiffusionPipeline, user: str, device: str):
    """
    Run the DDIM vs DDPM experiment using the loaded diffusion pipeline.
    Args:
        pipe (DiffusionPipeline): The loaded diffusion pipeline.
        user (str): The user for path management.
        device (str): The device to use for inference.
    """
    # Read prompts
    prompts = read_prompts(user, '10_prompts.txt')

    # Define output directory
    output_dir = f"outputs/ddim_vs_ddpm"
    os.makedirs(output_dir, exist_ok=True)

    # Define schedulers
    schedulers = {
        "ddim": DDIMScheduler.from_config(pipe.scheduler.config),
        "ddpm": DDPMScheduler.from_config(pipe.scheduler.config),
    }

    for scheduler_name, scheduler in schedulers.items():
        pipe.scheduler = scheduler
        pipe.to(device)

        for i, prompt in enumerate(prompts):
            image = pipe(prompt=prompt).images[0]

            filename = f"{output_dir}/{i:02d}_{scheduler_name}.png"
            image.save(filename)
            print(f"Saved: {filename}")
            
    print(f"Experiment completed. Images saved in {output_dir}.")