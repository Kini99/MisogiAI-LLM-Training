import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import os

def load_dataset(file_path):
    """Load the dataset from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return Dataset.from_list(data)

def tokenize_function(examples, tokenizer):
    """Tokenize the examples"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

def main():
    # Configuration
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dataset_path = "dataset.jsonl"
    output_dir = "./polite_helper_model"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens if they don't exist
    special_tokens = ["<|user|>", "<|assistant|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Resize token embeddings to account for new tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Move model to CPU
    model = model.to("cpu")
    
    print("Setting up LoRA configuration...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # Rank
        lora_alpha=32,  # Alpha parameter for LoRA scaling
        lora_dropout=0.1,  # Dropout probability for LoRA layers
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    print("Applying LoRA to model...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("Loading dataset...")
    dataset = load_dataset(dataset_path)
    
    print("Tokenizing dataset...")
    def tokenize_dataset(examples):
        return tokenize_function(examples, tokenizer)
    
    tokenized_dataset = dataset.map(
        tokenize_dataset,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,  # 3-5 epochs as specified
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,  # As specified
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        do_eval=True,
        save_strategy="steps",
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,  # Disable wandb/tensorboard logging
    )
    
    print("Setting up data collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    print("Setting up trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,  # Using same dataset for eval
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("Training completed!")
    print(f"Model saved to: {output_dir}")

if __name__ == "__main__":
    main() 