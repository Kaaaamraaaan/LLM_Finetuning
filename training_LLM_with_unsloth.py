import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import json
import sys
import pandas as pd
import torch
import tensorflow as tf
from datasets import load_dataset, DatasetDict
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

def special_token(text):
    text = text.replace('<s>', '<|begin_of_text|>')
    text = text.replace('</s>', '<|eot_id|>')
    return text

def chat_modification(text):
    return text.replace(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|>",
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nAs an expert in the field of energy, please answer the user's query.<|eot_id|>"
    ).replace('<|end_of_text|>', '')

def prompt_format_llama3(row, tokenizer):
    system_prompt = "As an expert in the field of energy, please answer the user's query."
    dialogue_template = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Please answer the following question.\nQuestion: {row['question']}"},
        {"role": "assistant", "content": row['answer']}
    ]
    prompt = tokenizer.apply_chat_template(
        conversation=dialogue_template,
        tokenize=False,
        add_generation_prompt=False
    )
    return prompt

def initialize_model_and_tokenizer(model_name, chosen_model):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        dtype=None,
        load_in_4bit=False,
        trust_remote_code=True,
    )

    for n, p in model.named_parameters():
        if p.device.type == 'meta':
            print(f"{n} is on meta!")

    if chosen_model in ["llama-2-13b-chat-hf", "llama-2-7b-chat-hf", "llama-2-7b-hf", "misteral-7b-instruct-v2"]:
        tokenizer.eos_token = '</s>'
    elif chosen_model in ["llama-3-8b", "llama-3-8b-instruct"]:
        tokenizer.eos_token = '<|eot_id|>'

    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "right"

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    assert model.config.pad_token_id == tokenizer.pad_token_id, "Pad token ID mismatch!"

    print('Tokenizer pad token ID:', tokenizer.pad_token_id)
    print('Model pad token ID:', model.config.pad_token_id)

    return model, tokenizer

def prepare_datasets(chosen_model, tokenizer):
    df_base = pd.read_json("./datasets/related_papers_books_wikis_final_version4.json", lines=True)
    if chosen_model in ["llama-3-8b", "llama-3-8b-instruct"]:
        df_base['Text'] = df_base['Text'].apply(special_token)

    df_QA = pd.read_json("./datasets/Q&A_for_RAG_embedding_V1_papers_books_54470_QA_labeled.json", lines=True)
    df_QA = df_QA[df_QA['Label'] == 'Good question'][['question', 'answer']].reset_index(drop=True)
    df_QA['Text'] = df_QA.apply(lambda row: prompt_format_llama3(row, tokenizer), axis=1)

    df_train, df_test = df_QA[:50000], df_QA[50000:]
    df_base = df_base[['Text']]
    df_train = pd.concat([df_base, df_train], ignore_index=True).sample(frac=1)

    return df_train, df_test

def train_model(model, tokenizer, df_train, df_test, chosen_model):
    train_path = f'./datasets/df_{chosen_model}_train_{len(df_train)}.jsonl'
    test_path = f'./datasets/df_{chosen_model}_test_{len(df_test)}.jsonl'

    df_train.to_json(train_path, orient='records', lines=True)
    df_test.to_json(test_path, orient='records', lines=True)

    train_data = load_dataset('json', data_files=train_path, split="train")
    test_data = load_dataset('json', data_files=test_path, split="train")

    data = DatasetDict({'train': train_data, 'test': test_data})

    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=True,
        modules_to_save=['embed_tokens', 'lm_head']
    )

    print('GPU 0:', torch.cuda.memory_allocated(0))

    training_config = {
        "learning_rate": 1e-6,
        "learning_rate_end": 1e-7,
        "weight_decay": 0.001,
        "max_grad_norm": 0.3,
        "logging_steps": 2000,
        "eval_steps": 2000,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "polynomial",
        "lr_scheduler_power": 1.0,
        "save_total_limit": 3,
        "num_train_epochs": 3,
    }

    output_dir = (
        f"./unsloth/{chosen_model}/"
        f"V-4-QA-LR-[{training_config['learning_rate']},{training_config['learning_rate_end']}]"
        f"-{training_config['lr_scheduler_type']}-power-{training_config['lr_scheduler_power']}"
        f"-weight_decay-{training_config['weight_decay']}-max_grad_norm-{training_config['max_grad_norm']}"
        f"-not-quantized-dropout-0.05/"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        optim="paged_adamw_32bit",
        do_eval=True,
        eval_strategy="steps",
        save_strategy="epoch",
        learning_rate=training_config['learning_rate'],
        logging_strategy="steps",
        logging_dir=output_dir,
        logging_steps=training_config['logging_steps'],
        max_grad_norm=training_config['max_grad_norm'],
        num_train_epochs=training_config['num_train_epochs'],
        max_steps=-1,
        eval_steps=training_config['eval_steps'],
        warmup_ratio=training_config['warmup_ratio'],
        weight_decay=training_config['weight_decay'],
        group_by_length=True,
        bf16=True,
        lr_scheduler_type=training_config['lr_scheduler_type'],
        lr_scheduler_kwargs={'lr_end': training_config['learning_rate_end'], 'power': training_config['lr_scheduler_power']},
        ddp_find_unused_parameters=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        save_total_limit=training_config['save_total_limit'],
        disable_tqdm=True,
        report_to="tensorboard"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=data['train'],
        eval_dataset=data['test'],
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="Text",
        packing=True,
        data_collator=None
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    with torch.autocast("cuda"):
        trainer.train()

def main():
    models = [
        'Bart-large', 'llama-2-13b-chat-hf', 'llama-2-7b-chat-hf',
        'llama-2-7b-hf', 'llama-3-8b', 'llama-3-8b-instruct',
        'misteral-7b-instruct-v2', 'phi-2', 'PHI-3-4k-context'
    ]
    chosen_model = models[5]
    model_name = f"./models/{chosen_model}/"

    model, tokenizer = initialize_model_and_tokenizer(model_name, chosen_model)
    df_train, df_test = prepare_datasets(chosen_model, tokenizer)
    print('Sample record:\n', df_train['Text'][len(df_train) // 2])
    train_model(model, tokenizer, df_train, df_test, chosen_model)

if __name__ == "__main__":
    main()
