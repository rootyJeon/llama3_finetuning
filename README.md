# Llama3_finetuning
This repo is a tutorial of Llama3.0 finetuning for Google DSC, Korea Univ. <br>
You can finetune Meta's open LLM, **Llama3-70B** with custom GPUs (24G VRAM)

## Efficiently scale distributed training
* Model: Llama 3-70B
* Strategy: PyTorch FSDP and Q-LoRA

## 1. Installation
You need to install Hugging Face libraries and Pyroch, including trl, transformers and datasets.
```
# Install PyTorch for FSDP and FA/SDPA
pip install "torch==2.2.2" tensorboard

# Install Hugging Face libraries
pip install --upgrade "transformers==4.40.0" "datasets==2.18.0" "accelerate==0.29.3" "evaluate==0.4.1" "bitsandbytes==0.43.1" "huggingface_hub==0.22.2" "trl==0.8.6" "peft==0.10.0"
```
You may also need to enroll your Hugging Face access tokens. Please refer the site: [Hugging Face's User Access Tokens](https://huggingface.co/docs/hub/security-tokens)
```
huggingface-cli login --token "[YOUR TOKEN]"
```

## 2. Prepare the dataset
We will use the [HuggingFaceH4/no_robots dataset](https://huggingface.co/datasets/HuggingFaceH4/no_robots), which has 10,000 (trainset: 9,500 and testset: 500) instructions and demonstrations. You can use this dataset for supervised fine-tuning (SFT) to make language models follow instructions better.
```json\n",
"{\"messages\": [{\"role\": \"system\", \"content\": \"You are...\"}, {\"role\": \"user\", \"content\": \"...\"}, {\"role\": \"assistant\", \"content\": \"...\"}]}",
"{\"messages\": [{\"role\": \"system\", \"content\": \"You are...\"}, {\"role\": \"user\", \"content\": \"...\"}, {\"role\": \"assistant\", \"content\": \"...\"}]}",
"{\"messages\": [{\"role\": \"system\", \"content\": \"You are...\"}, {\"role\": \"user\", \"content\": \"...\"}, {\"role\": \"assistant\", \"content\": \"...\"}]}",
...
```
The dataset will be downloaded by the *Datasets library*.

## 3. Finetune the Llama3 with PyTorch FSDP, Q-LoRA
Since our code is optimized for 4 GPUs, you need to adjust your settings. Also, you need to use ```torchrun``` for finetuning as we are running in a distributed training.
For ```torchrun``` and FSDP we need to set the environment variable ```ACCELERATE_USE_FSDP``` and ```FSDP_CPU_RAM_EFFICIENT_LOADING``` to tell transformers/accelerate to use FSDP and load the model in a memory-efficient way. <br>

*Note: To NOT CPU offloading you need to change the value of fsdp and remove offload. This only works on > 40GB GPUs since it requires more memory.* <br>

Now, start your finetuning with the following command 🚀:
```
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=4 ./scripts/run_fsdp_qlora.py --config llama_3_70b_fsdp_qlora.yaml
```
### Expected Memory usage:

* Full-finetuning with FSDP needs ~16*80GB GPUs
* FSDP + LoRA needs ~8*80GB GPUs
* FSDP + Q-Lora needs ~2*40GB GPUs
* FSDP + Q-Lora + CPU offloading needs 4*24GB GPUs, with 22 GB/GPU and 127 GB CPU RAM with a sequence length of 3072 and a batch size of 1.

