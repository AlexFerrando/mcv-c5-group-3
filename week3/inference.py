import wandb
import torch
from models import BaselineModel, LSTMModel
from our_tokenizers import CharacterTokenizer
from torchvision import transforms
from PIL import Image

# Per descarregar-lo des de wandb. Només cal fer-ho un cop.
# run = wandb.init()
# artifact = run.use_artifact('arnalytics-universitat-aut-noma-de-barcelona/C5-W3/baseline:v93', type='model')
# artifact_dir = artifact.download()

model_dir = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/artifacts/baseline:v93/baseline.pth'

tokenizer = CharacterTokenizer()
model = LSTMModel(tokenizer, lstm_layers=1, dropout=0.3)
model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def generate_caption(img: torch.Tensor, max_length: int = 201) -> str:
    with torch.no_grad():
        logits = model(img.to(device), teacher_forcing=False)
        return model.logits_to_text(logits)[0]


def predict(image_path: str) -> str:
    # Preprocesamiento compatible con ResNet
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)  # Añadir dimensión batch
    
    return generate_caption(img_tensor)

image = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C5. Visual Recognition/mcv-c5-group-3/week3/archive/Food Images/Food Images/-burnt-carrots-and-parsnips-56390131.jpg'
caption = predict(image)
print(f"Predicción: {caption}")
