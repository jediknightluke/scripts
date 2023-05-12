"""

Written by Luke Melton
03/01/2023

Fine-tuning GPT-2 and Generating Texts
--------------------------------------

This script fine-tunes a GPT-2 model on a 
custom dataset and generates texts using the fine-tuned model.



"""


import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class CustomTextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

        self.examples = []

        for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
            self.examples.append(tokenized_text[i:i + block_size])
        if len(tokenized_text) % block_size != 0:
            self.examples.append(tokenized_text[-block_size:])  # Add the last truncated tokenized_text

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def generate_text(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=50,
        num_return_sequences=5,
        no_repeat_ngram_size=2,
        top_k=50,  # Add this line
        do_sample=True,  # Add this line
    )
    generated_texts = []

    for output in outputs:
        generated_texts.append(tokenizer.decode(output, skip_special_tokens=True))

    return generated_texts

def fine_tune_gpt2(
    model_name="gpt2",
    data_dir="/opt/scripts/",
    output_dir="/opt/scripts/output/",
    num_train_epochs=3,
    train_batch_size=8,
    logging_steps=100,
    save_steps=10000,
    save_total_limit=2,
):
    # Tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Prepare the dataset
    def load_dataset(file_path):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return CustomTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=128)



    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    datasets = [load_dataset(file_path) for file_path in all_files]
    dataset = torch.utils.data.ConcatDataset(datasets)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # Train the model
    trainer.train()

    # Save the trained model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

# Example usage
fine_tune_gpt2(
    model_name="gpt2",
    data_dir="/opt/scripts/",
    output_dir="/opt/scripts/ML/",
    num_train_epochs=3,
    train_batch_size=8,
    logging_steps=100,
    save_steps=10000,
    save_total_limit=2,
)

output_dir = "/opt/scripts/ML/"
model = GPT2LMHeadModel.from_pretrained(output_dir)
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

prompt = "Once upon a time"
generated_texts = generate_text(prompt, model, tokenizer)

for text in generated_texts:
    print(text)
