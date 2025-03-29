import torch
import json
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import os

# Ensure the model directory exists
os.makedirs('./model', exist_ok=True)

class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = []
    labels = []
    tag2id = {intent['tag']: i for i, intent in enumerate(data['intents'])}
    
    for intent in data['intents']:
        texts.extend(intent['patterns'])
        labels.extend([tag2id[intent['tag']]] * len(intent['patterns']))
    
    return texts, labels, tag2id

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

def main():
    # Load and prepare data
    texts, labels, tag2id = load_data('intents_augmented.json')
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Tokenize
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    
    # Create datasets
    train_dataset = IntentDataset(train_encodings, train_labels)
    val_dataset = IntentDataset(val_encodings, val_labels)
    
    # Training setup
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(tag2id),
        id2label={i: tag for tag, i in tag2id.items()}
    )
    
  
    training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,  # Increased epochs
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,  # Optimal for fine-tuning
    eval_strategy="steps",  # Updated parameter name
    eval_steps=200,
    save_steps=200,
    logging_steps=50,
    warmup_ratio=0.1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=True,  # Enable if using GPU
    report_to="none"
    )    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train and save
    trainer.train()
    model.save_pretrained('./model')
    tokenizer.save_pretrained('./model')
    
    # Save label mappings
    with open('./model/tag2id.json', 'w', encoding='utf-8') as f:
        json.dump(tag2id, f, ensure_ascii=False)

if __name__ == "__main__":
    main()