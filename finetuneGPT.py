import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import AutoModelForSequenceClassification
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

from generatePrompts import generateList

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

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

bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModelForSequenceClassification.from_pretrained("./resultsCompleteFineTuning/checkpoint-4500")
bert_model.eval()
bert_model.requires_grad_(False)

prompts = generateList()

def get_reward(generated_texts):
    inputs = bert_tokenizer(generated_texts, return_tensors="pt", padding=True, truncation=True).to("mps")
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        return probs[:, 1]
    
prompts = generateList()

responses = []
for prompt in tqdm(prompts):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    responses.append((prompt, decoded))

def score_responses(pairs):
    texts = [resp for _, resp in pairs]
    inputs = reward_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = reward_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        rewards = probs[:, 1].cpu().numpy()
    return rewards

rewards = score_responses(responses)

filtered_data = [
    {"text": prompt + " " + response}
    for (prompt, response), r in zip(responses, rewards)
    if r > 0.8  # you can lower this if too few examples
]

print(f"Selected {len(filtered_data)} / {len(prompts)} responses with high reward.")

dataset = Dataset.from_list(filtered_data)

# --- Step 4: Fine-tune with LoRA on good responses ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./lora-ppo-conditioned",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(), 
    report_to="none"
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./lora-ppo-conditioned")
tokenizer.save_pretrained("./lora-ppo-conditioned")