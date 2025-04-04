from collections import OrderedDict
import os

from transformers import AutoModelForCausalLM, VisionEncoderDecoderModel, ViTModel, AutoModelForCausalLM, ViTConfig, ViTImageProcessor, AutoTokenizer
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
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
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

print("Configuring the decoder for encoder-decoder architecture...")
decoder.config.is_decoder = True
decoder.config.add_cross_attention = True


print("Loading the encoder-decoder model...")
model = VisionEncoderDecoderModel(encoder=vit_encoder, decoder=decoder)
model.to(DEVICE)  # Move the model to the correct device
print("Encoder-decoder model loaded successfully.")

if tokenizer.bos_token_id is not None:
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    
    
# Set pad token (important for batching and generation)
if tokenizer.pad_token_id is not None:
    model.config.pad_token_id = tokenizer.pad_token_id
else:
    model.config.pad_token_id = tokenizer.encode(tokenizer.bos_token)[-1]
# --- Backpropagation Test ---
print("\n--- Starting Backpropagation Test ---")
import torch.optim as optim # 1. Import the optimizer module

# 2. Define Optimizer instance (using AdamW here)
# Create AdamW optimizer:
# - It targets model.parameters() (all parameters in the model for this test)
# - Sets a learning rate (lr) of 1e-5 (0.00001)
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# 1. Set model to training mode
model.train()
print("Model set to training mode.")

# 2. Define Optimizer (Simple AdamW on all parameters for test)
#    Note: For actual PEFT tuning, filter parameters to optimize only adapters etc.
try:
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    print("Optimizer initialized (AdamW on all parameters for this test).")
except Exception as e:
    print(f"Error initializing optimizer: {e}")
    exit()

# 3. Create Dummy Inputs and Labels
batch_size = 2 # Use a slightly larger batch size to test batching
num_channels = 3
height = 224
width = 224
seq_length = 15 # Example sequence length for labels

# Dummy pixel values (Input to encoder)
pixel_values = torch.randn(batch_size, num_channels, height, width, dtype=torch.float32).to(DEVICE)
print(f"Dummy pixel_values created with shape: {pixel_values.shape}")

# Dummy labels (Target output for decoder)
# Ensure labels are within vocab size and use LongTensor
# Using tokenizer.eos_token_id as a valid token ID for simplicity
if tokenizer.eos_token_id is None:
     print("Error: Cannot create dummy labels without tokenizer.eos_token_id")
     exit()

# Create labels tensor filled with EOS token ID, shape (batch_size, seq_length)
labels = torch.full((batch_size, seq_length), tokenizer.eos_token_id, dtype=torch.long).to(DEVICE)
# Optionally, add some variation or specific start/end tokens if needed for a more realistic test
# labels[:, 0] = tokenizer.bos_token_id # If BOS is used
print(f"Dummy labels created with shape: {labels.shape}")


# 4. Perform Forward Pass with Labels, Calculate Loss, Backward Pass, Optimizer Step
print("Performing forward pass with labels, backward pass, and optimizer step...")
try:
    # Zero gradients before the forward pass
    optimizer.zero_grad()

    # Forward pass - provide both pixel_values and labels
    # The model calculates loss internally when labels are provided
    outputs = model(pixel_values=pixel_values, labels=labels)

    # Extract the loss
    loss = outputs.loss
    if loss is None:
        print("Error: Model did not return a loss. Check model's forward method and label passing.")
        exit()

    print(f"Forward pass successful. Loss: {loss.item()}") # Use .item() to get scalar value

    # Backward pass - compute gradients
    loss.backward()
    print("Backward pass successful (gradients computed).")

    # Optional: Check if gradients exist for some parameters
    # Example: Check gradient of a LoRA weight and a base model weight
    lora_param_name = None
    base_param_name = None
    for name, param in model.named_parameters():
         if "lora_" in name and param.requires_grad:
              lora_param_name = name
              break
    for name, param in model.named_parameters():
         if "decoder.model.layers.0.self_attn.q_proj.weight" in name: # Example base Llama param
              base_param_name = name
              break

    if lora_param_name:
         lora_param = dict(model.named_parameters())[lora_param_name]
         print(f"Gradient check - LoRA param '{lora_param_name}' grad exists: {lora_param.grad is not None}")
         # print(f"LoRA param grad norm: {lora_param.grad.norm().item() if lora_param.grad is not None else 'N/A'}")
    if base_param_name:
         base_param = dict(model.named_parameters())[base_param_name]
         # Base model grads should be None if frozen, or exist if unfrozen/included by optimizer
         print(f"Gradient check - Base param '{base_param_name}' grad exists: {base_param.grad is not None}")
         # print(f"Base param grad norm: {base_param.grad.norm().item() if base_param.grad is not None else 'N/A'}")


    # Optimizer step - update weights based on gradients
    optimizer.step()
    print("Optimizer step successful.")

except Exception as e:
    print(f"\n--- Error during backpropagation test ---")
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
    print("-----------------------------------------")
