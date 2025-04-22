from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor, PreTrainedTokenizer
from dataset import FoodDataset
from torch.utils.data import Subset

from torch.utils.data import random_split
import consts
import torch
import utils
import pickle

if __name__ == '__main__':

    # Load tokenizer, feature extractor and model
    MODEL_NAME = 'nlpconnect/vit-gpt2-image-captioning'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    feature_extractor = ViTImageProcessor.from_pretrained(MODEL_NAME)

    # Define dataset and create dataloaders
    dataset = FoodDataset(
        # consts.DATA_PATH,
        consts.DRINK_DATA_PATH,
        tokenizer,
        feature_extractor,
    )
    
    # Split in train, validation and test
    train_size, val_size, test_size = utils.get_split_sizes(len(dataset), 0.8, 0.1, 0.1)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(consts.SEED)
    )

    df = dataset.df
    condition = (df['Classification'] == 'drink') & (df['Annotation'] == 2)

    # 3. Find test indices that match the condition
    drink_test_indices = [i for i in test_dataset.indices if condition.iloc[i]]

    # 4. Create a subset from the original dataset using those indices
    test_dataset_only_drinks = Subset(dataset, drink_test_indices)
    
    with open('test_dataset_only_drinks.pkl', 'wb') as f:
        pickle.dump(test_dataset_only_drinks, f)
    
    drink_train_indices = [i for i in train_dataset.indices if condition.iloc[i]]
    train_dataset_only_drinks = Subset(dataset, drink_train_indices)

    with open('train_dataset_only_drinks.pkl', 'wb') as f:
        pickle.dump(train_dataset_only_drinks, f)
        
    drink_val_indices = [i for i in val_dataset.indices if condition.iloc[i]]
    val_dataset_only_drinks = Subset(dataset, drink_val_indices)
    with open('val_dataset_only_drinks.pkl', 'wb') as f:
        pickle.dump(val_dataset_only_drinks, f)
