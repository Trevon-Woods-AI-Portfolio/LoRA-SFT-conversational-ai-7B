from transformers import BitsAndBytesConfig, AutoModelForCausalLM, DataCollatorForLanguageModeling, AutoTokenizer
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.3",
    quantization_config=quantization_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.3",
    model_max_length=2048,
    truncation=True,
)

# REQUIRED for Mistral / LLaMA-style models
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
eval_dataset = load_dataset("timdettmers/openassistant-guanaco", split="test")


peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

sft_config = SFTConfig(
    "fine_tuned_conversational_ai",
    push_to_hub=True,
    per_device_train_batch_size=8,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    num_train_epochs=2,
    eval_strategy="steps",
    eval_steps=200,
    logging_steps=200,
    gradient_checkpointing=True,
    max_length=512,
    # New parameters
    dataset_text_field="text",
    packing=False,
    report_to="none",
)

trainer = SFTTrainer(
    model,
    args=sft_config,
    peft_config=peft_config,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
)

trainer.train()