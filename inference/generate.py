from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# load the base model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.3",
    torch_dtype=torch.float16,
    device_map="auto",
)

# load the adapter
model.load_adapter("AIStrong/fine_tuned_conversational_ai")

gen_conversation = pipeline("text-generation", model=model, tokenizer=tokenizer)
gen_conversation("### Human: Hello! ### Assistant:", max_new_tokens=100)