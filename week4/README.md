# Week 4: Image Captioning (II)

⚠️⚠️⚠️ Check our final slides [here](https://docs.google.com/presentation/d/17mf8qRFvCeGZUFsSpi91ZiALxGDk06jlI7QYSyJsF78/edit?slide=id.g31bab77bdae_0_0#slide=id.g31bab77bdae_0_0)!

📝📝📝 See the project paper [here](https://overleaf.cvc.uab.cat/read/tbtthcmqxmft#83be8e)!

Welcome to **Week 4** of our project, where we take **Image Captioning** a step further by leveraging **pre-trained multimodal models** and **parameter-efficient fine-tuning**. Building upon last week’s baseline and custom models, this week we explore:

- Finetuning a **pre-trained ViT-GPT2** model.
- Applying **PEFT (LoRA)** to adapt the model more efficiently.
- Evaluating zero-shot capabilities of models like **BLIP2**.

We continue to work on the [Food Ingredients and Recipes Dataset with Images](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images), aiming to generate dish titles from food images.

---

### 🍽️ Example:

<img src="https://assets.epicurious.com/photos/5f99a91e819b886aba0a2846/1:1/w_1920,c_limit/Chickensgiving_HERO_RECIPE_101920_1374_VOG_final.jpg" alt="Example Food Image" width="400"/>

**Desired Caption:** *"Miso-Butter Roast Chicken With Acorn Squash Panzanella"*

---

## 📂 Project Structure

```
├── consts.py                 # Constants and configuration settings
├── dataset.py                # Dataset class for main training/evaluation
├── dataset_task_2.py         # Dataset class for direct evaluation (BLIP2)
├── download_model.py         # Script to download pre-trained models
├── evaluator.py              # Evaluation metrics and scoring
├── plots.ipynb               # Notebook for plotting metrics/visualizations
├── qualitative_eval.py       # Script for qualitative evaluation
├── task1.py                  # Finetuning ViT-GPT2 model
├── task2_PEFT.py             # LoRA-based PEFT finetuning
├── task2_direct_eval.py      # BLIP2 direct evaluation (zero-shot)
├── task2_test.py             # Script for testing/evaluation
├── utils.py                  # Helper functions
```

---

## 🏋️‍♂️ Tasks Overview

### 🔧 Task 1: Finetuning ViT-GPT2

We fine-tune a **ViT-GPT2** model for image captioning using our food dataset. The model leverages powerful pre-trained vision and language components.

To run finetuning:

```bash
python task1.py
```

---

### 🔍 Task 2.1: Direct Evaluation with BLIP2

We evaluate the zero-shot captioning ability of **BLIP2**, a strong multimodal model, without further training.

To run direct evaluation:

```bash
python task2_direct_eval.py
```

---

### 🪶 Task 2.2: PEFT with LoRA

We apply **Low-Rank Adaptation (LoRA)** to fine-tune the model efficiently with fewer trainable parameters.

To run LoRA fine-tuning:

```bash
python task2_PEFT.py
```

---

## 🖼️ Dataset

We use the same dataset as Week 3: **Food Ingredients and Recipes Dataset with Images**. Make sure it's downloaded and properly linked.

🔗 **Dataset link:** [Kaggle Dataset](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images)

---

## 📊 Evaluation & Visualization

- Use `evaluator.py` to compute metrics like BLEU, METEOR, CIDEr, etc.
- Run `qualitative_eval.py` for visual comparison of predictions.
- Open `plots.ipynb` to explore and visualize training progress and results.

---

## 🚀 Setup Instructions

Make sure all required libraries are installed:

```bash
pip install -r requirements.txt
```

Then you’re ready to run training, evaluation, and testing scripts!

---

## 📌 Notes

- Modify paths and configurations as needed in `consts.py`.
- Check `download_model.py` to fetch fine tuned models from Wandb.
- LoRA-based fine-tuning is more memory-efficient — ideal for large-scale experiments.
- Zero-shot performance (BLIP2) offers a strong baseline for comparison.

---

For any questions or debugging help, reach out to the team! 🚀
