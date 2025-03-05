from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "state-spaces/mamba2-1.3b"
model = AutoModelForCausalLM.from_pretrain(model_name)
tokenizer = AutoTokenizer.from_pretrain(model_name)

from peft import find_all_linear_names

target_modules = find_all_linear_names(model)
print(target_modules)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    target_modules=target_modules,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)

model = get_peft_model(model, lora_config)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="path/to/output/dir",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epoch=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="linear",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Your prepared dataset
    data_collator=collator,  # Depending on your dataset
)

trainer.train()
model.save_pretrain("path/to/save/model")