# Finetuning LLMs for Text Generation

This repo was created for an assignment in my NLP/LLM class. The goal was to take pre-trained models and finetune them with a dataset to adapt them to generate text/code based on an instruction.
These were the models used:
 - https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main
 - https://huggingface.co/microsoft/phi-2/tree/main
 - https://huggingface.co/mistralai/Mistral-7B-v0.1/tree/main

## Requirements
 - A GPU with lots of RAM (40GB+)
 - Python 3.8+

Required Libraries/Modules made be installed with this command:
```
pip install accelerate peft bitsandbytes transformers trl rouge bert_score evaluate
```

## Dataset
https://huggingface.co/datasets/vicgalle/alpaca-gpt4

## Usage
There's a python file to finetune each pretrained model. Each program prints the finetuned model's first 20 responses to the terminal. Each program, after generating a finetuned model, runs 10 trials with the model, using different values of `top_k`, `num_beams`, and `temperature` for each trial. Each trial prints the model's responses for 20 inputs. 
### finetune llama-2
To use the llama2 model, you will first need to get and set your own huggingface auth token in `finetune_llama2.py`:
```
access_token = 'YOUR_HF_TOKEN'
```
Then you can run the following command:
```
python finetune_llama2.py
```
### finetune phi-2
```
python finetune_phi2.py
```
### finetune mistral
```
python finetune_mistral.py
```

## Trial Results
In the trials, there are 4 different values used for `temperature`, `top_k`, and `num_beams`:
 - `temperature`: 1.00, 0.66, 0.33, 0.00
 -  `top_k`: 8, 4, 2, 1
 -  `num_beams`: 8, 4, 2, 1
### mistral
Here I observed when `num_beams` was set to 1, the other params had no visible effects: 
```
Params								BLEU		Rouge		BERT
Temperature: 1.00, Top_K: 1, Num_beam: 1			0.0557		0.5922		0.7330
Temperature: 1.00, Top_K: 1, Num_beam: 2			0.0557		0.4887		0.7422
Temperature: 1.00, Top_K: 1, Num_beam: 4			0.0568		0.5327		0.7411
Temperature: 1.00, Top_K: 1, Num_beam: 8			0.0589		0.6632		0.7355
Temperature: 1.00, Top_K: 2, Num_beam: 1			0.0557		0.5922		0.7330
Temperature: 1.00, Top_K: 4, Num_beam: 1			0.0557		0.5922		0.7330
Temperature: 1.00, Top_K: 8, Num_beam: 1			0.0557		0.5922		0.7330
Temperature: 0.66, Top_K: 1, Num_beam: 1			0.0557		0.5922		0.7330
Temperature: 0.33, Top_K: 1, Num_beam: 1			0.0557		0.5922		0.7330
Temperature: 0.01, Top_K: 1, Num_beam: 1			0.0557		0.5922		0.7330
```

### phi-2
```
Params								BLEU		Rouge		BERT
Temperature: 1.00, Top_K: 8, Num_beam: 8			0.1718		0.6624		0.6898
Temperature: 1.00, Top_K: 8, Num_beam: 4			0.1269		0.6305		0.7167
Temperature: 1.00, Top_K: 8, Num_beam: 2			0.0792		0.5168		0.7098
Temperature: 1.00, Top_K: 8, Num_beam: 1			0.0911		0.5378		0.7020
Temperature: 1.00, Top_K: 4, Num_beam: 8			0.1718		0.6624		0.6898
Temperature: 1.00, Top_K: 2, Num_beam: 8			0.1718		0.6624		0.6898
Temperature: 1.00, Top_K: 1, Num_beam: 8			0.1718		0.6624		0.6898
Temperature: 0.66, Top_K: 8, Num_beam: 8			0.1718		0.6624		0.6898
Temperature: 0.33, Top_K: 8, Num_beam: 8			0.1718		0.6624		0.6898
Temperature: 0.01, Top_K: 8, Num_beam: 8			0.1718		0.6624		0.6898
```

### llama-2
```
Params								BLEU		Rouge		BERT
Temperature: 1.00, Top_K: 8, Num_beam: 8			0.0562		0.6818		0.7401
Temperature: 1.00, Top_K: 8, Num_beam: 4			0.0567		0.6366		0.7407
Temperature: 1.00, Top_K: 8, Num_beam: 2			0.0699		0.6727		0.7384
Temperature: 1.00, Top_K: 8, Num_beam: 1			0.0658		0.3748		0.7495
Temperature: 1.00, Top_K: 4, Num_beam: 8			0.0579		0.7058		0.7361
Temperature: 1.00, Top_K: 2, Num_beam: 8			0.0560		0.6698		0.7385
Temperature: 1.00, Top_K: 1, Num_beam: 8			0.0560		0.6698		0.7385
Temperature: 0.66, Top_K: 8, Num_beam: 8			0.0565		0.6678		0.7370
Temperature: 0.33, Top_K: 8, Num_beam: 8			0.0571		0.6939		0.7372
Temperature: 0.01, Top_K: 8, Num_beam: 8			0.0560		0.6698		0.7385
```

## Model Metric Comparison
```
Model Name		BLEU		Rouge		BERT
llama-2			0.0560		0.6698		0.7385
phi-2			0.1718		0.6624		0.6898
mistral			0.0589		0.6632		0.7355
```

## References
The code wis partially based on this datacamp tutorial: https://www.datacamp.com/tutorial/fine-tuning-llama-2
