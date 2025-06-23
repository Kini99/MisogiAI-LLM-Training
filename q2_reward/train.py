import pandas as pd
from datasets import Dataset
from itertools import combinations
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from trl import RewardTrainer

def create_comparison_dataset(df: pd.DataFrame):
    """
    Create a dataset of comparisons from a dataframe with prompts, answers, and ranks.
    For each prompt, create pairs of (chosen, rejected) answers based on their ranks.
    """
    pairs = []
    for _, group in df.groupby('prompt'):
        for (idx1, row1), (idx2, row2) in combinations(group.iterrows(), 2):
            prompt = row1['prompt']
            if row1['rank'] < row2['rank']:
                chosen = row1['answer']
                rejected = row2['answer']
            elif row2['rank'] < row1['rank']:
                chosen = row2['answer']
                rejected = row1['answer']
            else:
                continue  # Skip pairs with the same rank

            pairs.append({'chosen': f"{prompt} {chosen}", 'rejected': f"{prompt} {rejected}"})
            
    return Dataset.from_list(pairs)

def train_reward_model():
    """
    Train and save a reward model.
    """
    try:
        df = pd.read_csv('q2_reward/answers.csv')
    except FileNotFoundError:
        print("Error: `q2_reward/answers.csv` not found.")
        print("Please create `answers.csv` with 'prompt', 'answer', and 'rank' columns, and manually rank the answers.")
        return

    if not all(col in df.columns for col in ['prompt', 'answer', 'rank']):
        print("Error: `answers.csv` must contain 'prompt', 'answer', and 'rank' columns.")
        return

    # Create the comparison dataset
    comparison_dataset = create_comparison_dataset(df)

    # Load a tokenizer and a model
    model_name = "distilbert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(text=examples["chosen"], text_pair=examples["rejected"], truncation=True, padding='max_length', max_length=512)

    tokenized_dataset = comparison_dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./reward_model_trainer_output',
        per_device_train_batch_size=4,
        num_train_epochs=1, # We use max_steps instead
        max_steps=75,
        learning_rate=2e-5,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none", # "tensorboard" if you have it installed
    )

    # Create a dummy evaluation dataset if you want to see evaluation metrics during training
    # For simplicity, we can just re-use a fraction of the training data as eval data
    shuffled_dataset = tokenized_dataset.shuffle(seed=42)
    eval_dataset = shuffled_dataset.select(range(int(len(shuffled_dataset) * 0.1)))


    # Instantiate the RewardTrainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset,
        eval_dataset=eval_dataset, # Optional
    )

    # Train the model
    print("Training reward model...")
    trainer.train()
    print("Training finished.")

    # Save the model
    print("Saving model to `q2_reward/reward_model/`...")
    trainer.save_model('q2_reward/reward_model')
    tokenizer.save_pretrained('q2_reward/reward_model')
    print("Model saved successfully.")

if __name__ == "__main__":
    train_reward_model() 