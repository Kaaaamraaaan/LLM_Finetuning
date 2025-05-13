import torch
import transformers
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from trl import SFTTrainer
from accelerate import PartialState
import pandas as pd
import json
import time


def special_token(text):
    text = text.replace(
        '<s>',
        '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    )
    text = text.replace('</s>', '<|eot_id|><|end_of_text|>')
    return text


def main():
    start_time = time.time()
    device_map = "DDP"

    if device_map == "DDP":
        device_string = PartialState().process_index
        device_map = {'': device_string}

    model_name = "./models/llama-3-8b-instruct/"

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=bnb_config,
        cache_dir='',
        use_cache=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=True
    )
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "right"
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    assert model.config.pad_token_id == tokenizer.pad_token_id, \
        "The model's pad token ID does not match the tokenizer's pad token ID!"

    print('Tokenizer pad token ID:', tokenizer.pad_token_id)
    print('Model pad token ID:', model.config.pad_token_id)

    df = pd.read_json(
        "./datasets/related_papers_books_wikis_final_version2.json",
        lines=True, orient='records'
    )
    df2 = pd.read_json(
        "./datasets/related_prompt_to_renewable_geothermal_energy_from_HuggingFaceH4_ultrachat_200k_for_llama3.json",
        lines=True, orient='records'
    )

    df['Text'] = df['Text'].apply(special_token)
    df = df[['Text']]
    concatenated_df = pd.concat([df, df2], ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    print('Sample text:\n', concatenated_df['Text'][round(len(concatenated_df) / 2)])

    concatenated_df.to_json(
        '/home/kamran.haddadian1/datasets/df_llama_3.jsonl',
        orient='records', lines=True
    )
    data = load_dataset('json', data_files='/home/kamran.haddadian1/datasets/df_llama_3.jsonl', split="train")

    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["input_layernorm", "post_attention_layernorm", "norm"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print('GPU 0:', torch.cuda.memory_allocated(0))
    print('GPU 1:', torch.cuda.memory_allocated(1))

    training_args = transformers.TrainingArguments(
        output_dir="/home/kamran.haddadian1/saved_finetunedmodel_llama-3/everything-version-2-thirdtry-learning-rate1e-5linear-parallel/",
        per_device_train_batch_size=10,
        gradient_accumulation_steps=3,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_dir="/home/kamran.haddadian1/logging_info_llama-3/everything-version-2-thirdtry-learning-rate1e-5-linear-parallel/",
        learning_rate=1e-5,
        max_grad_norm=0.3,
        num_train_epochs=5,
        max_steps=-1,
        warmup_ratio=0.05,
        group_by_length=True,
        lr_scheduler_type="linear",
        ddp_find_unused_parameters=True,
        save_total_limit=15,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        peft_config=lora_config,
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="Text",
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    with torch.autocast("cuda"):
        trainer.train()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
