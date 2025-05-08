import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoModelForSequenceClassification
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import numpy as np
from generatePrompts import generateList, generateRealisticPrompts
import time

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)


reward_model = AutoModelForSequenceClassification.from_pretrained("./resultsCompleteFineTuning/checkpoint-4500").to(device)
reward_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
reward_model.eval()

def get_reward(generated_texts):
    batch_size = 8
    rewards = []

    for i in range(0, len(generated_texts), batch_size):
        batch = generated_texts[i:i+batch_size]
        inputs = reward_tokenizer(batch, return_tensors = 'pt', padding=True, truncation=True).to("mps")
        with torch.no_grad():
            outputs = reward_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            rewards.extend(probs[:, 1].cpu().numpy())

    return np.array(rewards)


def score_responses(pairs):
    texts = [resp for _, resp in pairs]
    inputs = reward_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = reward_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        rewards = probs[:, 1].cpu().numpy()
    return rewards

def tokenize(example):
    encoded = tokenizer(example["text"],
                        truncation = True,
                        padding = "max_length",
                        max_length = 512
            )
    encoded["labels"] = encoded["input_ids"].copy()
    return encoded

def generate_responses(prompts, model, batch_size=4):
    responses = []
    for i in range(0, len(prompts), batch_size):
        print(f"We are on batch {i}")
        start_time = time.time()

        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).to(device)
        output_ids = model.generate(**inputs, max_new_tokens=256, temperature = 0.9, top_p = 0.95, do_sample=True, pad_token_id = tokenizer.eos_token_id)
        batch_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens = True)
        responses.extend(zip(batch_prompts, batch_outputs))

        end_time = time.time()
        print(f"{end_time - start_time} seconds")

    return responses
    

epochs = 5
for epoch in range(epochs):

    test_prompts = generateRealisticPrompts(1000)
    responses = generate_responses(test_prompts, model)

    rewards = get_reward([r for _, r in responses])
    high_quality_data = [{"text": r} for (p, r), score in zip(responses, rewards) if score > 0.7]

    print(high_quality_data)
    print(f"[Epoch {epoch+1}] Selected {len(high_quality_data)} / {len(test_prompts)} responses with reward > 0.8")

    if len(high_quality_data) < 5:
        print(f"[Epoch {epoch+1}] Skipping fine-tuning due to insufficient high-reward samples.")
        continue

    dataset = Dataset.from_list(high_quality_data)
    tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./lora-ppo-conditioned",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    trainer.train()

    eval_prompts = generateRealisticPrompts(100)
    eval_responses = generate_responses(eval_prompts, model)
    eval_rewards = get_reward([r for _, r in eval_responses])
    mean_reward = float(np.mean(eval_rewards))
    print(f"[Epoch {epoch+1}] Mean eval reward: {mean_reward:.4f}")


    


model.save_pretrained("./lora-ppo-conditioned")
tokenizer.save_pretrained("./lora-ppo-conditioned")