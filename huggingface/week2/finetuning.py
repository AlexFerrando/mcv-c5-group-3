import consts

import albumentations as A
from finetuning_utils import augment_and_transform_batch, collate_fn
from inference import load_model
from transformers import TrainingArguments, Trainer
from read_data import VideoDataset
from functools import partial

DATA_PATH = consts.KITTI_MOTS_PATH

# Define training arguments
training_args = TrainingArguments(
    output_dir="./outputs/alex/mask2former_finetuned",
    num_train_epochs=75,
    fp16=False,
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8, 
    dataloader_num_workers=4,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    weight_decay=1e-4,
    max_grad_norm=0.01,
    metric_for_best_model="eval_map",
    greater_is_better=True,
    load_best_model_at_end=False, # True if we want to load the best model at the end of training
    eval_strategy="epoch",
    eval_steps=1,
    save_strategy="epoch",
    save_total_limit=1,
    remove_unused_columns=False,
    eval_do_concat_batches=False,
    push_to_hub=False,
    logging_dir="./outputs/alex/logs",
    logging_steps=25,
)


if __name__ == '__main__':
    video_dataset = VideoDataset(DATA_PATH)
    data = video_dataset.load_data()
    # data_dict = video_dataset.split_data(data) # In order to have train, test, and validation sets
    data_dict = data.train_test_split(test_size=0.2)

    model, image_processor = load_model()

    train_augment_and_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
    )
    validation_transform = A.Compose(
        [A.NoOp()],
    )

    # Make transform functions for batch and apply for dataset splits
    train_transform_batch = partial(
        augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
    )
    validation_transform_batch = partial(
        augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
    )

    train_data = data_dict["train"].with_transform(train_transform_batch)
    val_data = data_dict["train"].with_transform(validation_transform_batch)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=image_processor,
        data_collator=collate_fn,
    )

    trainer.train()