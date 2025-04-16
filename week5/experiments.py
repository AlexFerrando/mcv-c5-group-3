import consts
import time
import torch
import os
from diffusers import DDIMScheduler, DDPMScheduler, DiffusionPipeline

SEED = 42

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

    pipe.to(device)
    for scheduler_name, scheduler in schedulers.items():
        pipe.scheduler = scheduler
        pipe.scheduler.set_timesteps(20)  # Set the number of denoising steps

        total_time = 0.0

        for i, prompt in enumerate(prompts):
            torch.manual_seed(SEED)
            generator = torch.Generator(device=device).manual_seed(SEED)

            start_time = time.time()
            image = pipe(prompt=prompt, generator=generator, num_inference_steps=20).images[0]
            end_time = time.time()

            elapsed = end_time - start_time
            total_time += elapsed

            filename = f"{output_dir}/{i:02d}_{scheduler_name}.png"
            image.save(filename)
            print(f"Saved: {filename} (Time: {elapsed:.2f}s)")

        avg_time = total_time / len(prompts)
        print(f"Average execution time per prompt for {scheduler_name.upper()}: {avg_time:.4f} seconds")

    print(f"Experiment completed. Images saved in {output_dir}.")


def run_experiment_num_denoising_steps(pipe: DiffusionPipeline, user: str, device: str):
    """
    Run the experiment with different denoising steps using the loaded diffusion pipeline.
    Args:
        pipe (DiffusionPipeline): The loaded diffusion pipeline.
        user (str): The user for path management.
        device (str): The device to use for inference.
    """
    # Read prompts
    prompts = read_prompts(user, '10_prompts.txt')

    # Define output directory
    output_dir = f"outputs/num_denoising_steps"
    os.makedirs(output_dir, exist_ok=True)

    # Define denoising steps
    denoising_steps = [1, 5, 10, 20, 50, 100]

    for steps in denoising_steps:
        pipe.scheduler.set_timesteps(steps)  # Set the number of denoising steps
        pipe.to(device)

        total_time = 0
        for i, prompt in enumerate(prompts):
            torch.manual_seed(SEED)
            generator = torch.Generator(device=device).manual_seed(SEED)

            start_time = time.time()
            image = pipe(prompt=prompt, num_inference_steps=steps, generator=generator).images[0]
            end_time = time.time()

            elapsed = end_time - start_time
            total_time += elapsed

            filename = f"{output_dir}/{i:02d}_{steps}_steps.png"
            image.save(filename)
            print(f"Saved: {filename} (Time: {elapsed:.2f}s)")

        avg_time_per_step = total_time / len(prompts)
        print(f"Average execution time for {steps} steps: {avg_time_per_step:.4f} seconds")

    print(f"Experiment completed. Images saved in {output_dir}.")


def run_negative_prompt_experiment(pipe: DiffusionPipeline, user: str, device: str):
    """
    Run an experiment comparing outputs with and without negative prompts.
    Args:
        pipe (DiffusionPipeline): The loaded diffusion pipeline.
        user (str): The user for path management.
        device (str): The device to use for inference.
    """
    # Read prompts
    prompts = read_prompts(user, '10_prompts.txt')

    # Define a negative prompt
    negative_prompt = "blurry, distorted, low quality, bad anatomy, deformed, watermark"

    # Define output directory
    output_dir = "outputs/negative_prompting"
    os.makedirs(output_dir, exist_ok=True)

    pipe.to(device)

    for i, prompt in enumerate(prompts):
        torch.manual_seed(SEED)
        generator = torch.Generator(device=device).manual_seed(SEED)
        pipe.scheduler.set_timesteps(20)  # Set the number of denoising steps

        # With negative prompt
        image_with_neg = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=20  # Adjust as needed
        ).images[0]
        filename_with = f"{output_dir}/{i:02d}_with_negative.png"
        image_with_neg.save(filename_with)
        print(f"Saved: {filename_with}")

        # Without negative prompt
        generator = torch.Generator(device=device).manual_seed(SEED)  # Reset generator
        image_without_neg = pipe(
            prompt=prompt,
            negative_prompt=None,
            generator=generator,
            num_inference_steps=20  # Adjust as needed
        ).images[0]
        filename_without = f"{output_dir}/{i:02d}_without_negative.png"
        image_without_neg.save(filename_without)
        print(f"Saved: {filename_without}")

    print(f"Negative prompt experiment completed. Results saved in {output_dir}.")



def run_cfg_strength_experiment(pipe: DiffusionPipeline, user: str, device: str):
    """
    Run an experiment testing the effect of different CFG (guidance_scale) strengths.
    Args:
        pipe (DiffusionPipeline): The loaded diffusion pipeline.
        user (str): The user for path management.
        device (str): The device to use for inference.
    """
    # Read prompts
    prompts = read_prompts(user, '10_prompts.txt')

    # Define CFG scales to test
    cfg_scales = [1.0, 3.5, 7.5, 12.0]

    # Define output directory
    output_dir = "outputs/cfg_strength"
    os.makedirs(output_dir, exist_ok=True)

    pipe.to(device)

    for scale in cfg_scales:
        total_time = 0.0

        for i, prompt in enumerate(prompts):
            torch.manual_seed(SEED)
            generator = torch.Generator(device=device).manual_seed(SEED)

            start_time = time.time()
            image = pipe(
                prompt=prompt,
                guidance_scale=scale,
                generator=generator
            ).images[0]
            end_time = time.time()

            elapsed = end_time - start_time
            total_time += elapsed

            filename = f"{output_dir}/{i:02d}_cfg_{scale:.1f}.png"
            image.save(filename)
            print(f"Saved: {filename} (Time: {elapsed:.2f}s)")

        avg_time = total_time / len(prompts)
        print(f"Average execution time per prompt for CFG={scale}: {avg_time:.4f} seconds")

    print(f"CFG strength experiment completed. Results saved in {output_dir}.")


def run_model_comparison(pipe: DiffusionPipeline, user: str, device: str):
    """
    Run the experiment generating high-quality realistic food images using the loaded diffusion pipeline.
    Args:
        pipe (DiffusionPipeline): The loaded diffusion pipeline.
        user (str): The user for path management.
        device (str): The device to use for inference.
    """
    # Read prompts
    prompts = read_prompts(user, '10_prompts.txt')

    # Get model name for output directory
    model_name = pipe.config._name_or_path.split("/")[-1].replace(" ", "_")

    # Define output directory
    output_dir = f"outputs/model_comparison/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Standard aspect ratios: 512x512, 768x512, or 1024x768 typically work well
    width = 512
    height = 512

    pipe.to(device)

    # Adjustable parameters for quality
    num_inference_steps = 50  

    for i, prompt in enumerate(prompts):
        
        # Set seed for reproducibility (comment out for random results each time)
        generator = torch.Generator(device=device).manual_seed(42)
        
        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]

        filename = f"{output_dir}/{i:02d}_.png"
        image.save(filename)
        print(f"Saved image: {filename}")

    print(f"Experiment completed. Images saved in {output_dir}.")


def run_experiment_food(pipe: DiffusionPipeline, user: str, device: str):
    """
    Run the experiment generating high-quality realistic food images using the loaded diffusion pipeline.
    Args:
        pipe (DiffusionPipeline): The loaded diffusion pipeline.
        user (str): The user for path management.
        device (str): The device to use for inference.
    """
    # Read prompts
    prompts = read_prompts(user, 'food.txt')

    # Improved base prompt for ultra-realistic food photography
    base_prompt = (
        "award-winning professional food photography, shot with Canon EOS R5, 85mm f/1.4 lens, "
        "natural soft window lighting from the left, studio environment, shallow depth of field, "
        "4K resolution, high detail textures, smooth bokeh background, crisp focus on the food, "
        "commercial quality, magazine cover worthy, hyperrealistic, photorealistic"
    )
    
    # Negative prompt to avoid common AI generation artifacts
    negative_prompt = (
        "cartoon, illustration, drawing, painting, anime, collage, watermark, text, blurry, "
        "deformed, disfigured, bad anatomy, extra limbs, extra fingers, mutation, oversaturated, "
        "low quality, low resolution, artificial, computer generated"
    )

    # Get model name for output directory
    model_name = pipe.config._name_or_path.split("/")[-1].replace(" ", "_")

    # Define output directory
    output_dir = f"outputs/model_comparison/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Higher resolution for better details - adjust based on GPU capability
    # Standard aspect ratios: 512x512, 768x512, or 1024x768 typically work well
    width = 512
    height = 512

    pipe.to(device)

    # Adjustable parameters for quality
    num_inference_steps = 100  # More steps for higher quality
    guidance_scale = 7.5      # Controls adherence to prompt (7-9 works well for realism)
    num_variations = 2        # Generate multiple variations of each prompt

    for i, prompt in enumerate(prompts):
        # Enhanced prompt engineering with context and specific details
        food_item = prompt.strip()
        full_prompt = (
            f"Exquisite {food_item} presented on a designer ceramic plate, garnished with fresh herbs, "
            f"droplets of sauce artfully placed, professional food styling, {base_prompt}"
        )
        
        # Consider different angles and compositions
        if i % 3 == 0:  # Vary composition for diversity
            full_prompt += ", top-down view, flat lay composition"
        elif i % 3 == 1:
            full_prompt += ", 45-degree angle view, showing depth and layers"
        else:
            full_prompt += ", side view displaying layers and textures"

        print(f"Generating: {food_item}")
        
        # Generate multiple variations
        for v in range(num_variations):
            # Set seed for reproducibility (comment out for random results each time)
            seed = 1000 * i + v
            generator = torch.Generator(device=device).manual_seed(seed)
            
            image = pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]

            filename = f"{output_dir}/{i:02d}_{food_item.replace(' ', '_')}_{v}_100.png"
            image.save(filename)
            print(f"  Saved variation {v+1}: {filename}")

    print(f"Experiment completed. Images saved in {output_dir}.")
