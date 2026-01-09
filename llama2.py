import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

from huggingface_hub import snapshot_download

from datasets import Dataset
import numpy as np
from pathlib import Path

def load_and_tokenize_data(file_path, tokenizer, max_length=512):
    """
    Загрузка и токенизация данных из текстового файла
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Разбиваем текст на примеры
    examples = []
    for i in range(0, len(text), max_length):
        chunk = text[i:i + max_length]
        if len(chunk) >= 100:  # Отбросит последний чанк
            examples.append(chunk)
    
    # Токенизация
    tokenized = tokenizer(
        examples,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    return tokenized


# Конфигурация
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
DATA_FILE = "text.txt"
OUTPUT_DIR = "./llama-fine-tuned"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

print("Загрузка токенизатора...")
model_id = "meta-llama/Llama-2-7b-hf"
snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# Добавляем pad token если его нет
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Загрузка модели...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)


if not torch.cuda.is_available():
    model.to(device)

print("Загрузка и токенизация данных...")
tokenized_data = load_and_tokenize_data(DATA_FILE, tokenizer, max_length=512)


dataset_dict = {
    'input_ids': tokenized_data['input_ids'],
    'attention_mask': tokenized_data['attention_mask']
}

# Добавляем labels для языкового моделирования
dataset_dict['labels'] = tokenized_data['input_ids'].clone()

dataset = Dataset.from_dict(dataset_dict)

# Разделение на train/validation (80/20)
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# Data collator для языкового моделирования
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Аргументы обучения
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    logging_steps=50,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    save_strategy="steps",
    learning_rate=2e-5,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",  
    save_total_limit=2,
)

# Создание Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Начало обучения
print("Начало дообучения...")
train_result = trainer.train()

# Сохранение результатов
print("Сохранение модели...")
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)

# Логирование результатов
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

print("Дообучение завершено!")
print(f"Модель сохранена в: {OUTPUT_DIR}")
