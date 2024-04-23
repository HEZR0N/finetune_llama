# 2024 Hezron E Perez
import os
from huggingface_hub.hf_api import HfFolder
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

import nltk
from rouge import Rouge
from bert_score import BERTScorer
from evaluate import load

# Model from Hugging Face hub
base_model = "microsoft/phi-2"

# New instruction dataset
alpaca_dataset = "vicgalle/alpaca-gpt4"

# Fine-tuned model
new_model = "phi-2-finetuned"

# Get dataset
dataset = load_dataset(alpaca_dataset, split="train")
# Remove entries with an input (ie if instruction is "Tell me about this article", the input would be the article)
dataset = dataset.filter(lambda example: example['input'] == '')
# Select a subset of the data, because out GPU has limited RAM
dataset = dataset.select(range(20))

# Use quantization to reduce memory usage and improve inference speed
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Load pre-trained model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0},
)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Test the original model
test_text = dataset['instruction'][0]
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200, top_k=1, num_beams=2, temperature=0.5)
result = pipe(f"<s>[INST] {test_text} [/INST]")
print("Pre-trained Model Answer: ", result[0]['generated_text'])

# Train a new model by finetuning
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)


training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.train()

trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)



# Reload model in FP16 and merge it with LoRA weights
load_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)

new_test_model = PeftModel.from_pretrained(load_model, new_model)
new_test_model = new_test_model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Test the new model
test_text = dataset['instruction'][0]

default_pipe = pipeline(task="text-generation", model=new_test_model, tokenizer=tokenizer, max_length=200,
    top_k=1, num_beams=1, temperature=1)
result = default_pipe(f"<s>[INST] {test_text} [/INST]")
print("Original Model Answer: ", result[0]['generated_text'])


# Get Metrics for finetuned model
nltk.download('punkt')
nltk.download('bleu')
rouge = Rouge()
bertscore = load("bertscore")
trial_num = 0
trials = []

# Runs pipe and calculates metrics to return
def get_metrics(pipe, blue, red, bert, ground_truth):
    new_res = pipe(f"<s>[INST] {ground_truth} [/INST]")
    new_res = new_res[0]['generated_text']
    blue.append(nltk.translate.bleu_score.sentence_bleu([ground_truth.split()], new_res.split()))
    red.append(rouge.get_scores(new_res, ground_truth))
    raw_bert = bertscore.compute(predictions=[new_res], references=[ground_truth], model_type="distilbert-base-uncased")
    bert.append(raw_bert['f1'][0])
    return new_res, blue, red, bert

# Creates new pipe with given args. calls get_metrics(). Saves metrics to array. 
def test_pipe(temp, top_k, num_beam):
    global trial_num
    trial_num += 1
    trial_params = f"Temperature: {temp:.2f}, Top_K: {top_k}, Num_beam: {num_beam}"
    print(f"Start Trial {trial_num}: {trial_params}")

    BLEU = []
    Rouge = []
    BERTScore = []
        
    pipe = pipeline(task="text-generation", model=new_test_model, tokenizer=tokenizer, max_length=200,
        top_k=top_k, num_beams=num_beam, temperature=temp)

    # Loop to test dataset    
    for i in range(20):
        cur_text = dataset['instruction'][i]
        if i < 20:
            print("Q: ", cur_text)
        result, BLEU, Rouge, BERTScore = get_metrics(pipe, BLEU, Rouge, BERTScore, cur_text)
        if i < 20:
            print("A: ", result)

    print(f"End Trial {trial_num}: {trial_params}")
    # print(BLEU)
    blue = sum(BLEU)/len(BLEU)
    print("Average BLEU Score", blue)
    # print(Rouge)
    red = sum(i[0]['rouge-l']['f'] for i in Rouge)/len(Rouge)
    print("Average Rouge Score", red)
    # print(BERTScore)
    bert = sum(BERTScore)/len(BERTScore)
    print("Average BERTScore", bert)
    trials.append([trial_params, float(blue), float(red), float(bert)])


# Test Finetuned Model: Default Pipeline
test_pipe(1, 8, 8)

# Testing Different Beam Values
test_pipe(1, 8, 4)

test_pipe(1, 8, 2)

test_pipe(1, 8, 1)

# Testing Different Top_K Values
test_pipe(1, 4, 8)

test_pipe(1, 2, 8)

test_pipe(1, 1, 8)

# Testing Different Tempurature Values
test_pipe(0.66, 8, 8)

test_pipe(0.33, 8, 8)

test_pipe(0.01, 8, 8)

# Print Summary
print("Params\t\t\t\t\t\t\t\tBLEU\t\tRouge\t\tBERT")
for i in range(10):
    t = trials[i]
    print(f"{t[0]}\t\t\t{t[1]:.4f}\t\t{t[2]:.4f}\t\t{t[3]:.4f}")
