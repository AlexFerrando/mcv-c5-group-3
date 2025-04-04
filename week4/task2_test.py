from collections import OrderedDict
import os

from transformers import AutoModelForCausalLM, VisionEncoderDecoderModel, ViTModel, AutoModelForCausalLM, ViTConfig, ViTImageProcessor
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
import torch

login(os.environ['HF_TOKEN'])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

feature_extractor = ViTImageProcessor.from_pretrained('nlpconnect/vit-gpt2-image-captioning')


print("Loading the encoder weights from the ViT model...")
# Load the encoder weights from the ViT model
vit_finetuned_path = "/ghome/c5mcv03/mcv-c5-group-3/archive/artifacts/fine-tune-both:v0/best_model.pth"
finetuned_model = torch.load(vit_finetuned_path, map_location=DEVICE)


encoder_only_model = OrderedDict((k.replace("encoder.", "", 1), v) for k, v in finetuned_model.items() if k.startswith('encoder'))
vit_encoder = ViTModel(ViTConfig())
vit_encoder.load_state_dict(encoder_only_model)
print("Encoder weights loaded successfully.")

print("Loading the decoder weights from the Llama model...")
# Load the decoder model
decoder = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')

# Define LoRA configuration
lora_config = LoraConfig(
    r=16,  # Rank of the update matrices
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.05,  # Dropout probability
    bias="none",  # Bias type
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target modules to apply LoRA
    task_type="CAUSAL_LM"  # Task type
)

# Apply LoRA to the decoder
decoder = get_peft_model(decoder, lora_config)
print("PEFT applied succesfully.")

print("Loading the encoder-decoder model...")
model = VisionEncoderDecoderModel(encoder=vit_encoder, decoder=decoder)
print("Encoder-decoder model loaded successfully.")
print(model)