import consts

from finetuning_utils import transform, collate_fn
from inference import load_model
from transformers import TrainingArguments, Trainer
from read_data import VideoDataset
from functools import partial

DATA_PATH = consts.KITTI_MOTS_PATH_ALEX

# Define training arguments
training_args = TrainingArguments(
    output_dir="./outputs/alex/mask2former_finetuned",
    num_train_epochs=75,
    fp16=False,
    per_device_train_batch_size=1, 
    per_device_eval_batch_size=1, 
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

video_dataset = VideoDataset(DATA_PATH)
data = video_dataset.load_data()
# data_dict = video_dataset.split_data(data) # In order to have train, test, and validation sets
data_dict = data.train_test_split(test_size=0.2)

model, image_processor = load_model()

train_transform = partial(transform, image_processor=image_processor)

train_data = data_dict["train"].with_transform(train_transform)
test_data = data_dict["test"].with_transform(train_transform)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    processing_class=image_processor,
    data_collator=collate_fn,
)

trainer.train()