from datasets import disable_caching, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from peft import get_peft_model, LoraConfig, TaskType
import torch 

def create_trainer(finetuning_method, tokenized_datasets):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model.to(device)

    if(finetuning_method == "Normal"):
        print("Normal Run!")
    if(finetuning_method == "Classification_Head"):
        print("Just the Classification Head!")
        for param in model.distilbert.parameters():
           param.requires_grad = False
    if(finetuning_method == "LoRA"):
        print("LoRA training!")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode = False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules = ["q_lin", "v_lin"]
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    training_args = TrainingArguments(
    output_dir="./resultsClassificationHead",
    eval_strategy = "epoch",
    learning_rate = 2e-5,
    per_device_eval_batch_size=16,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir = "./logs"
    )

    trainer = Trainer(
    model = model, 
    args = training_args,
    train_dataset = tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics
    )

    
    return trainer


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


if __name__ == "__main__":
    disable_caching()

    dataset = load_dataset("Pulk17/Fake-News-Detection-dataset")
    dataset = dataset["train"].train_test_split(test_size=0.2)
    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(example):
        texts = [t + " " + x for t, x in zip(example["title"], example["text"])]
        return tokenizer(texts, truncation=True, padding="max_length")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format("torch", columns = ['input_ids', 'attention_mask', 'label'])  
    
    trainer = create_trainer("Classification_Head", tokenized_datasets)
    trainer.train()











