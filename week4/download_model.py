import wandb

wandb.login(key='8410a2da3f323633210ca8d25ce6862368d9f489')

# Initialize a W&B run (ensure you’re logged in or have set your API key)
run = wandb.init(project="C5-W4", entity="arnalytics-universitat-aut-noma-de-barcelona")

# Access the artifact by specifying its name and optionally its version (e.g., 'model:latest')
artifact = run.use_artifact('arnalytics-universitat-aut-noma-de-barcelona/C5-W4/best_model:v0')

# Download the artifact locally; this returns the path to the downloaded files
artifact_dir = artifact.download()

print(f"Artifact downloaded to: {artifact_dir}")
