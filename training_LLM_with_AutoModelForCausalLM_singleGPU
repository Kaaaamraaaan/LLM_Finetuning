import argparse
import json
import sys
import time

import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig,
)
from trl import SFTTrainer

#  Configuration  #
MODEL_PATH = "./models/llama-3-8b-instruct/"
CACHE_DIR = ""
PAD_TOKEN = "<pad>"

DATA_PATH_1 = "./datasets/related_papers_books_wikis_final_version4.json"
OUTPUT_JSON_PATH = "./datasets/df_llama_3.jsonl"

LOGGING_BASE = "./logging_info_llama-3/"
MODEL_OUTPUT_BASE = "./saved_finetunedmodel_llama-3/"

TRAIN_PARAMS = {
    "epochs": 4,
    "learning_rate": 1e-6,
    "learning_rate_end": 5e-7,
    "weight_decay": 0.001,
    "max_grad_norm": 5.0,
    "lr_scheduler_type": "polynomial",
    "lr_scheduler_power": 2,
    "warmup_ratio": 0.05,
    "save_total_limit": 15,
    "optim": "paged_adamw_32bit"
}

#  Helper Functions  #

def replace_special_tokens(text):
    text = text.replace('<s>', '<|start_header_id|>assistant<|end_header_id|>\n\n')
    text = text.replace('</s>', '<|end_of_text|>')
    return text

def prepare_dataset():
    df = pd.read_json(DATA_PATH, lines=True, orient='records')
    df["Text"] = df["Text"].apply(replace_special_tokens)
    df = df[["Text"]]
    df.to_json(OUTPUT_JSON_PATH, orient="records", lines=True)
    print("Sample record:", df['Text'][len(df) // 2])
    return load_dataset("json", data_files=OUTPUT_JSON_PATH, split="train")

def build_output_dirs():
    if TRAIN_PARAMS["lr_scheduler_type"] == "polynomial":
        suffix = f"V-4-LR-[{TRAIN_PARAMS['learning_rate']},{TRAIN_PARAMS['learning_rate_end']}]-{TRAIN_PARAMS['lr_scheduler_type']}-power-{TRAIN_PARAMS['lr_scheduler_power']}"
    else:
        suffix = f"V-4-LR-{TRAIN_PARAMS['learning_rate']}-{TRAIN_PARAMS['lr_scheduler_type']}"

    suffix += f"-weight_decay-{TRAIN_PARAMS['weight_decay']}-max_grad_norm-{TRAIN_PARAMS['max_grad_norm']}-not-quantized/"
    
    return LOGGING_BASE + suffix, MODEL_OUTPUT_BASE + suffix

#  Load Model and Tokenizer  #

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=True)
tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    use_cache=False,
    cache_dir=CACHE_DIR,
)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
assert model.config.pad_token_id == tokenizer.pad_token_id

print("Tokenizer pad token ID:", tokenizer.pad_token_id)
print("Model pad token ID:", model.config.pad_token_id)

#  LoRA Config  #

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules="all-linear",
    modules_to_save=["lm_head"],
    bias="none",
    task_type="CAUSAL_LM",
)

#  Dataset  #

dataset = prepare_dataset()

#  Training Arguments  #

logging_dir, output_dir = build_output_dirs()

training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    optim=TRAIN_PARAMS["optim"],
    save_strategy="epoch",
    logging_strategy="epoch",
    logging_dir=logging_dir,
    log_level="info",
    learning_rate=TRAIN_PARAMS["learning_rate"],
    max_grad_norm=TRAIN_PARAMS["max_grad_norm"],
    num_train_epochs=TRAIN_PARAMS["epochs"],
    max_steps=-1,
    warmup_ratio=TRAIN_PARAMS["warmup_ratio"],
    weight_decay=TRAIN_PARAMS["weight_decay"],
    group_by_length=True,
    lr_scheduler_type=TRAIN_PARAMS["lr_scheduler_type"],
    lr_scheduler_kwargs={
        "lr_end": TRAIN_PARAMS["learning_rate_end"],
        "power": TRAIN_PARAMS["lr_scheduler_power"]
    },
    ddp_find_unused_parameters=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    save_total_limit=TRAIN_PARAMS["save_total_limit"],
    disable_tqdm=True,
)

#  Trainer  #

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field="Text",
    packing=True,
)

# Ensure normalization layers use float32
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module.to(torch.float32)

#  Train  #

print("GPU 0 Memory Usage:", torch.cuda.memory_allocated(0))
with torch.autocast("cuda"):
    trainer.train()

torch.cuda.empty_cache()
