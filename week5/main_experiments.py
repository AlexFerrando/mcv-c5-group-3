import torch
import experiments

from diffusers import DiffusionPipeline

def getargs():
    import argparse
    parser = argparse.ArgumentParser(description="Run Stable Diffusion")
    parser.add_argument(
        '--experiment', '-e', type=str, required=True, help='Experiment to run',
        choices=['ddim_vs_ddpm', 'num_denoising_steps', 'negative_prompting', 'cfg_strength', 'model_comparison'],
    )
    parser.add_argument(
        '--model', '-m', type=str, help='Model to use for inference',
        choices=['2.1', '2.1_turbo', 'xl', 'xl_turbo', '3.5_m'], default='xl',
    )
    parser.add_argument(
        '--user', '-u', type=str, default='server', help='User for path management',
        choices=['alex', 'arnau', 'pol', 'oriol', 'server']
    )
    parser.add_argument(
        '--device', '-d', type=str, default='cuda', help='Device to use for inference',
        choices=['cuda', 'cpu']
    )
    args = parser.parse_args()
    return args


def load_pipeline(model: str, seed: int = 42, device: str = 'cuda') -> DiffusionPipeline:
    """
    Load the diffusion pipeline for the specified model.
    Args:
        model (str): The model name to load.
    Returns:
        DiffusionPipeline: The loaded diffusion pipeline.
    """
    model_to_pipe = {
        '2.1': "stabilityai/stable-diffusion-2-1",
        '2.1_turbo': "stabilityai/sd-turbo",
        'xl': "stabilityai/stable-diffusion-xl-base-1.0",
        'xl_turbo': "stabilityai/sdxl-turbo",
        '3.5_m': "stabilityai/stable-diffusion-3.5-medium"
    }

    if model not in model_to_pipe:
        raise ValueError(f"Model {model} not found in available models.")
    
    return DiffusionPipeline.from_pretrained(model_to_pipe[model]).to(device)


def run_experiment(pipe: DiffusionPipeline, experiment: str, user: str, device: str):
    """
    Run the specified experiment using the loaded diffusion pipeline.
    Args:
        pipe (DiffusionPipeline): The loaded diffusion pipeline.
        experiment (str): The name of the experiment to run.
    """
    experiment_to_func = {
        'ddim_vs_ddpm': experiments.run_ddim_vs_ddpm_experiment,
        'num_denoising_steps': experiments.run_experiment_num_denoising_steps,
        'model_comparison': experiments.run_model_comparison,
        'negative_prompting': experiments.run_negative_prompt_experiment,
        'cfg_strength': experiments.run_cfg_strength_experiment,
    }

    if experiment not in experiment_to_func:
        raise ValueError(f"Experiment {experiment} not found in available experiments.")
    
    experiment_to_func[experiment](pipe, user, device)


if __name__ == "__main__":
    args = getargs()

    # Load the pipeline
    pipe = load_pipeline(args.model, args.device)
    run_experiment(pipe, args.experiment, args.user, args.device)
