from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForRLHFTraining
import torch
import random
import tqdm
import csv 
from datasets import Dataset

from generatePrompts import generateList

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForRLHFTraining.from_pretrained("gpt2").to("mps")
ref_model = AutoModelForRLHFTraining.from_pretrained("gpt2").to("mps")

from transformers import GenerationConfig
model.generation_config = GenerationConfig.from_pretrained("gpt2")
ref_model.generation_config = GenerationConfig.from_pretrained("gpt2")

bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModelForSequenceClassification.from_pretrained("./resultsCompleteFineTuning/checkpoint-4500")
bert_model.eval()
bert_model.requires_grad_(False)

def get_reward(generated_texts):
    inputs = bert_tokenizer(generated_texts, return_tensors="pt", padding=True, truncation=True).to("mps")
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        return probs[:, 1]
    
prompts = generateList()
train_dataset = Dataset.from_dict({"prompt": prompts})
    
config = PPOConfig(
    batch_size = 4,
    num_ppo_epochs = 4,
    learning_rate = 1e-5,
    cliprange = 0.2,
    kl_coef = 0.05
)

ppo_trainer = PPOTrainer(
    args=config,
    processing_class=tokenizer,
    model=model,
    ref_model=ref_model,
    reward_model=bert_model,
    train_dataset=train_dataset
)

reward_log = []

for _ in tqdm.tqdm(range(1000)):
    batch_prompts = random.sample(prompts, k=config.batch_size)
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("mps")
    output_ids = model.generate(**inputs, max_new_tokens=50)
    responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)


    rewards = get_reward(responses)
    reward_log.append(rewards.cpu().numpy().tolist())

    print("Sample reward batch:", rewards)
    ppo_trainer.step(batch_prompts, responses, rewards)


with open("reward_log.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "reward_1", "reward_2", "reward_3", "reward_4"])
    for i, row in enumerate(reward_log):
        writer.writerow([i] + row)
ppo_trainer.model.save_pretrained("./gpt2-ppo-fake-news")
tokenizer.save_pretrained("./gpt2-ppo-fake-news")