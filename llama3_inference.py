import torch
from random import randint
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


peft_model_id = "./llama-3-70b-hf-no-robot"

# Load Model with PEFT adapter
model = AutoPeftModelForCausalLM.from_pretrained(
  peft_model_id,
  torch_dtype=torch.float16,
  quantization_config= {"load_in_4bit": True},
  device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

# Load our test dataset
eval_dataset = load_dataset("json", data_files="test_dataset.json", split="train")
#messages = [{'content': 'Do you know Google Developer Student Clubs?', 'role': 'user'}]
messages = [{'content': 'Do you know Korea University?', 'role': 'user'}]

# Test on sample
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device, non_blocking=True) # GPU-efficient!!
outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    eos_token_id= tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]

print(f"**Query:**\n{messages[0]['content']}\n")
print(f"**Generated Answer:**\n{tokenizer.decode(response,skip_special_tokens=True)}")



# for i in range(3):
#     rand_idx = randint(0, len(eval_dataset))
#     messages = eval_dataset[rand_idx]["messages"][:2]
#     # Test on sample
#     input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device, non_blocking=True)
#     outputs = model.generate(
#         input_ids,
#         max_new_tokens=512,
#         eos_token_id= tokenizer.eos_token_id,
#         do_sample=True,
#         temperature=0.6,
#         top_p=0.9,
#     )
#     response = outputs[0][input_ids.shape[-1]:]

#     print(f"**Query:**\n{eval_dataset[rand_idx]['messages'][1]['content']}\n")
#     print(f"**Original Answer:**\n{eval_dataset[rand_idx]['messages'][2]['content']}\n")
#     print(f"**Generated Answer:**\n{tokenizer.decode(response,skip_special_tokens=True)}")
