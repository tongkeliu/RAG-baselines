from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import os
import json

def tokenize_function(examples):
    encodings = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

model_name = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

data_dir = "/root/autodl-pub/datasets/Amazon-review/sampled_reviews"
filename = "AE22QOZDLAODPKJHYZI2HXQ7MLZQ"
with open(os.path.join(data_dir, filename), "r") as f:
    selected_reviews = json.load(f)
texts = [r["text"] for r in selected_reviews]
dataset = Dataset.from_dict({"text": texts})
tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./lora-book-review",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="no",
    optim="adamw_torch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
model.save_pretrained("./models/lora-finetuned")

# test
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
content = tokenizer.decode(output_ids, skip_special_tokens=True)

print(content) 