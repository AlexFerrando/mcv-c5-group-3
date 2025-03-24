# Week 3: Image Captioning (I)

⚠️⚠️⚠️ Check our final slides [here](https://docs.google.com/presentation/d/1rZsFNczXgs0ZNEDZWPDb83qw1RwR42B14yrA6DDos6A/edit#slide=id.g340d9e9dd17_0_361)!

📝📝📝 See the paper for the project [here](https://overleaf.cvc.uab.es/project/67dff51c85f1f209c7a4c396)

Welcome to **Week 3** of our project, where we focus on **Image Captioning**. This repository contains all the necessary scripts and modules to train and evaluate captioning models for food images.
We will train and evaluate our models in the [Food Ingredients and Recipes Dataset with Images](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images). Objective will be to be able to generate the title from a food image. An example can be found here:

### 🍽️ Example:

<img src="https://assets.epicurious.com/photos/5f99a91e819b886aba0a2846/1:1/w_1920,c_limit/Chickensgiving_HERO_RECIPE_101920_1374_VOG_final.jpg" alt="Example Food Image" width="400"/>

**Desired Caption:** *"Miso-Butter Roast Chicken With Acorn Squash Panzanella"*

## 📂 Project Structure

```
├── tokenizers/                 # Experiments with different tokenizers
├── Baseline Model and Metrics.ipynb # Notebook with baseline experiments and metric evaluation
├── consts.py                   # Constants used throughout the project
├── dataset.py                  # Dataset class to read the FoodDataset
├── inference.py                 # Generate captions for images using trained models
├── metrics.py                   # Metric implementations for evaluation (e.g., BLEU, METEOR, CIDEr)
├── models.py                    # Implementation of different captioning models
├── our_tokenizers.py            # Custom tokenizer implementations
├── sweep_arnau.py               # Hyperparameter tuning script
├── test.ipynb                   # Testing notebook for inference & visualization
├── train.py                     # Main training script
└── tokenizers updated/          # Additional tokenizer experiments
```

---

## 🏋️‍♂️ Training the Model

The main training script is **`train.py`**, which handles the training process for different captioning models. To train a model, simply run:

```bash
python train.py
```

Ensure you have all dependencies installed before running the script.

---

## 📖 Dataset

The dataset used in this project is **FoodDataset**. The `dataset.py` file contains the `FoodDataset` class, which is responsible for loading and preprocessing the data.

🔗 **Dataset link:** (https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images)

---

## 🏗️ Models Implemented

The `models.py` file contains different architectures used in this project:

- **BaselineModel** – A simple baseline model.
- **LSTMModel** – A model using LSTMs for sequence generation.
- **LSTMWithAttention** – An advanced model incorporating attention mechanisms.

---

## 🖼️ Running Inference

To generate captions for an image using a trained model, modify the paths in `inference.py` and run it modifiying the paths in the code for the one desired.

```bash
python inference.py
```

This will generate a caption based on the trained model.

---

## 🔤 Tokenizers

Inside the `tokenizers/` folder, we have isolated experiments for different tokenizers. The implemented tokenizers in `our_tokenizers.py` include:

- **Character-level tokenizer**
- **BERT Tokenizer (WordPiece-level)**
- **Word-level tokenizer**

These tokenizers can be used to preprocess captions before training.

---

## 🚀 Installation & Requirements

Make sure you have all required dependencies installed:

```bash
pip install -r requirements.txt
```

Then, you’re all set to train and test your image captioning models! 🎉

---

## 📌 Notes
- Ensure your dataset is properly downloaded and linked.
- Modify configurations in `consts.py` for hyperparameter tuning.
- Experiment with different tokenizers to improve performance.

For any questions, feel free to reach out! 🚀

