---
base_model: Qwen/Qwen2.5-1.5B
library_name: peft
pipeline_tag: text-generation
language:
  - en
tags:
  - lora
  - peft
  - plain-language
  - easy-read
  - accessibility
  - text-simplification
---

# Qwen2.5-1.5B Easy Read LoRA

A LoRA adapter for [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) that rewrites official or complex English as short, plain sentences in an Easy Read style.

## Example

Input:

> Failure to remit the outstanding balance within fourteen days of the date of this notice may result in the council applying to the magistrates' court for a liability order, which may incur additional costs.

Output:

> You must pay the money you owe.\
> You must pay it within 14 days.\
> If you do not pay, the council can go to court.\
> The court can order you to pay more money.\
> This is called a liability order.\
> A liability order can cost you more money.

## How to use

The adapter expects the source text followed by a blank line and `Easy Read:` on its own line. It was trained to complete that prompt.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen2.5-1.5B"
adapter_id = "bentdixon/qwen2.5-1.5b-easy-read-lora"

tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(model, adapter_id)
model.eval()

text = ("Failure to remit the outstanding balance within fourteen days of the date of this notice "
        "may result in the council applying to the magistrates' court for a liability order, "
        "which may incur additional costs.")
prompt = f"{text}\n\nEasy Read:\n"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, do_sample=False, max_new_tokens=400,
                         pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
```

Greedy decoding (`do_sample=False`) is recommended. It runs on CPU, Apple Silicon (MPS) or CUDA.

## Training setup

| Setting | Value |
| --- | --- |
| Base model | Qwen/Qwen2.5-1.5B (base, not instruct) |
| Method | LoRA with PEFT 0.21, task `CAUSAL_LM` |
| Rank / alpha / dropout | 16 / 32 / 0.05 |
| Adapted layers | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable weights | About 18.5 million (roughly 1.2% of the base model) |
| Prompt | `{source}\n\nEasy Read:\n` |
| Loss | On the Easy Read completion only; the prompt is masked |

## Licence

The base model, Qwen2.5-1.5B, is released by the Qwen team under Apache 2.0. Check the licence of the training data before reusing this adapter commercially.
