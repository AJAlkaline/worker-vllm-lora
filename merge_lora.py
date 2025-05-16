#!/usr/bin/env python3
import argparse, os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",     required=True)
    p.add_argument("--lora_adapter",   required=True)
    p.add_argument("--output_dir",     default="/models")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Download base 4-bit quantized
    base_cache = snapshot_download(
        repo_id=args.base_model,
        cache_dir="/tmp/base",
        allow_patterns=["*.safetensors","*.bin","*.pt"]
    )
    # 2) Load tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tok.save_pretrained(args.output_dir)

    # 3) Load base in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        base_cache,
        load_in_4bit=True,
        device_map="auto",
        quantization_config=None,
        trust_remote_code=True
    )

    # 4) Apply LoRA adapter
    model = PeftModel.from_pretrained(model, args.lora_adapter, device_map="auto")
    # 5) Save merged model
    model.save_pretrained(args.output_dir)
    print(f"✅ Merged model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
