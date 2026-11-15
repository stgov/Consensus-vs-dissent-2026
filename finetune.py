import os
import math
import torch
import re
import json
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
from transformers import DataCollatorForSeq2Seq

# ---------------------------------------------------------------------------
# Colab setup (run these in notebook cells BEFORE this script):
#   !pip install -q -U transformers peft trl datasets bitsandbytes accelerate
#   from google.colab import drive; drive.mount('/content/drive')
#   Upload train_v5_3.jsonl and test_v5_2.jsonl to /content/ (or set the
#   env vars TRAIN_PATH / TEST_PATH).
#   Hugging Face token: add it as Colab secret "HF_TOKEN" (needed for Llama,
#   which is gated), or set the HF_TOKEN env var.
# ---------------------------------------------------------------------------

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token is None:
    try:
        from google.colab import userdata
        _hf_token = userdata.get("HF_TOKEN")
    except Exception:
        _hf_token = None
if _hf_token:
    login(token=_hf_token)

LOCAL_TRAIN_PATH = os.environ.get("TRAIN_PATH", "/content/train_v5_3.jsonl")
LOCAL_TEST_PATH = os.environ.get("TEST_PATH", "/content/test_v5_2.jsonl")

# Choose the model with MODEL=llama or MODEL=olmo (default: olmo).
# Paper (Sec. 3.1.3): Llama-3.1-8B-Instruct and OLMo-3-7B-Instruct-SFT,
# both with lr 1e-4, LoRA r=16 / alpha=32, 4 epochs, batch 2 x grad accum 4.
# NOTE: the original script used Olmo-3-7B-Think-SFT and the README says
# lr 2e-5 for Olmo; the paper text says Instruct-SFT and 1e-4 for both.
_MODELS = {
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "olmo": "allenai/Olmo-3-7B-Instruct-SFT",
}
_MODEL_KEY = os.environ.get("MODEL", "olmo").lower()
# Overrides, e.g. LEARNING_RATE=2e-5 to follow the README instead of the paper.
_LR = float(os.environ.get("LEARNING_RATE", "1e-4"))
# Saving to Drive keeps checkpoints if Colab disconnects.
# SMOKE=1: quick end-to-end check (64 train / 16 test examples, 20 optimizer steps).
# Not meant to produce a useful model; it only verifies the pipeline runs.
_SMOKE = os.environ.get("SMOKE", "0") == "1"
_OUT = os.environ.get(
    "OUTPUT_DIR",
    f"/content/drive/MyDrive/judgmental_{_MODEL_KEY}" + ("_smoke" if _SMOKE else ""),
)


@dataclass
class FineTuneConfig:

    model_name: str = _MODELS[_MODEL_KEY]

    train_path: str = LOCAL_TRAIN_PATH
    test_path: str = LOCAL_TEST_PATH

    output_dir: str = _OUT

    num_train_epochs: int = 1 if _SMOKE else 4
    max_steps: int = 20 if _SMOKE else -1   # -1 = train for all epochs
    batch_size: int = 2
    grad_accum_steps: int = 4
    learning_rate: float = _LR
    max_seq_length: int = 1275


    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    logging_steps: int = 5 if _SMOKE else 50
    save_steps: int = 10 if _SMOKE else 100
    eval_steps: int = 10 if _SMOKE else 100



def create_weighted_labels(input_ids, tokenizer, model_name):
    """
    label masking function (custom)
    """
    labels = [-100] * len(input_ids)

    if "olmo" in model_name.lower():
        # OLMo-3:
        # <|im_start|>assistant\n
        # <|im_end|>

        assistant_header = "<|im_start|>assistant\n"
        assistant_pattern = tokenizer.encode(assistant_header, add_special_tokens=False)

        skip_after_pattern = 0


        assistant_end_pattern = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        eot_token_id = tokenizer.eos_token_id

    else:
        # Llama: <|start_header_id|>assistant<|end_header_id|>\n\n
        assistant_header = "<|start_header_id|>assistant<|end_header_id|>"
        assistant_pattern = tokenizer.encode(assistant_header, add_special_tokens=False)
        skip_after_pattern = 0
        eot_token_id = tokenizer.eos_token_id

    start_idx = -1
    pattern_len = len(assistant_pattern)

    for i in range(len(input_ids) - pattern_len + 1):
        match = True
        for j in range(pattern_len):
            if input_ids[i + j] != assistant_pattern[j]:
                match = False
                break
        if match:
            start_idx = i + pattern_len + skip_after_pattern
            break

    if start_idx == -1:
        return labels


    if "olmo" not in model_name.lower():
        whitespace_tokens = set()
        for ws_char in ["\n", " ", "\r", "\t", "\n\n", "  "]:
            try:
                ws_toks = tokenizer.encode(ws_char, add_special_tokens=False)
                if ws_toks:
                    whitespace_tokens.update(ws_toks)
            except:
                pass

        if tokenizer.pad_token_id is not None:
            whitespace_tokens.add(tokenizer.pad_token_id)

        while start_idx < len(input_ids) and input_ids[start_idx] in whitespace_tokens:
            start_idx += 1

    end_idx = len(input_ids)

    if eot_token_id is not None:
        for i in range(start_idx, len(input_ids)):
            if input_ids[i] == eot_token_id:
                end_idx = i
                break

    for i in range(start_idx, end_idx):
        labels[i] = input_ids[i]

    return labels

import torch




def preprocess_function(examples, tokenizer, config):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            #enable_thinking=True ### ONLY FOR QWEN3
        )

        texts.append(text)


    model_inputs = tokenizer(
        texts,
        max_length=config.max_seq_length,
        truncation=True,
        padding=False,
    )


    labels = []
    for input_ids in model_inputs["input_ids"]:
        label = create_weighted_labels(input_ids, tokenizer, config.model_name)
        labels.append(label)

    model_inputs["labels"] = labels
    return model_inputs





def parse_output(output_string: str) -> dict:
    """
    Parse model output that may contain reasoning followed by ratings.
    """
    output_string = output_string.strip()

    try:
        cleaned = re.sub(r'```json\s*|\s*```', '', output_string)
        parsed = json.loads(cleaned)

        result = {}
        for key in ['fairness', 'Fairness']:
            if key in parsed:
                result['Fairness'] = parsed[key]
                break

        for key in ['consensus', 'Consensus']:
            if key in parsed:
                result['Consensus'] = parsed[key]
                break

        for key in ['satisfaction', 'Satisfaction']:
            if key in parsed:
                result['Satisfaction'] = parsed[key]
                break

        if len(result) == 3 and all(isinstance(v, (int, float)) for v in result.values()):
            result = {k: int(v) for k, v in result.items()}
            if all(-3 <= v <= 3 for v in result.values()):
                return result
    except (json.JSONDecodeError, AttributeError, ValueError, KeyError):
        pass


    pattern = re.compile(
    r'\*{0,2}(Fairness|Consensus|Satisfaction)\*{0,2}\s*\*{0,2}:\*{0,2}\s*(-?\d+)',
    re.IGNORECASE
)


    results = {}
    matches = list(pattern.finditer(output_string))

    for match in reversed(matches):
        key = match.group(1).capitalize()
        if key not in results:
            try:
                value = int(match.group(2))
                if -3 <= value <= 3:
                    results[key] = value
                else:
                    print(f"WARNING: Value {value} for {key} out of range [-3, 3]")
            except ValueError:
                print(f"WARNING: Could not convert '{match.group(2)}' to int for {key}")


    required_keys = ['Fairness', 'Consensus', 'Satisfaction']
    if len(results) != 3:
        missing = [k for k in required_keys if k not in results]
        print(f"WARNING: Found only {len(results)}/3 metrics. Missing: {missing}")
        print(f"   Input: {output_string[:150]}...")

    return results


def load_and_prepare_data(config):
    print("="*80)
    print("LOADING DATASETS")
    print("="*80)

    data = load_dataset(
        "json",
        data_files={
            "train": config.train_path,
            "test": config.test_path
        }
    )

    if _SMOKE:
        data["train"] = data["train"].select(range(min(64, len(data["train"]))))
        data["test"] = data["test"].select(range(min(16, len(data["test"]))))
        print("*** SMOKE MODE: 64 train / 16 test examples, max 20 steps ***")

    print(f"Train examples: {len(data['train'])}")
    print(f"Test examples: {len(data['test'])}")
    print(f"\nSample train example:")
    print(json.dumps(data["train"][0], indent=2))

    return data


def setup_model_and_tokenizer(config):
    print("\n" + "="*80)
    print("LOADING MODEL AND TOKENIZER")
    print("="*80)

    print(f"Loading tokenizer from {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,
        trust_remote_code=True,

    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        raise ValueError("Tokenizer missing chat_template!")

    print(f"✓ Tokenizer loaded")
    print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype
    )


    print(f"\nLoading model {config.model_name} with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=compute_dtype,
    )

    model = prepare_model_for_kbit_training(model)




    print(f"✓ Model loaded")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total params: {total_params:,}")

    return model, tokenizer


def formatting_func(example, tokenizer):
    """
    Format examples using chat template
    """
    texts = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        #enable_thinking=True ## ONLY QWEN
    )

    if isinstance(texts, str):
        texts = [texts]

    for i, text in enumerate(texts):
        if not text.endswith(tokenizer.eos_token):
            texts[i] = text + tokenizer.eos_token

    return texts


def train_model(config):
    """Main training function"""


    data = load_and_prepare_data(config)


    model, tokenizer = setup_model_and_tokenizer(config)


    print("\n" + "="*80)
    print("CONFIGURING TRAINING")
    print("="*80)

    # `warmup_ratio` was removed in transformers 5.x, so compute warmup steps explicitly
    # (10% of total steps, same as the original warmup_ratio=0.1; works on any version).
    _steps_per_epoch = math.ceil(len(data["train"]) / (config.batch_size * config.grad_accum_steps))
    _total_steps = config.max_steps if config.max_steps > 0 else _steps_per_epoch * config.num_train_epochs
    _warmup_steps = max(1, int(0.1 * _total_steps))
    # bf16 needs an Ampere+ GPU (A100/L4); fall back to fp16 on a T4.
    _use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    print(f"Total optimizer steps: {_total_steps} | warmup: {_warmup_steps} | precision: {'bf16' if _use_bf16 else 'fp16'}")

    training_args = TrainingArguments(
        output_dir=config.output_dir,


        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=config.grad_accum_steps,
        #eval_accumulation_steps=1,

        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=_warmup_steps,


        optim="paged_adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=0.3,
        gradient_checkpointing=True,


        fp16=not _use_bf16,
        bf16=_use_bf16,

        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,

        eval_strategy="steps",
        eval_steps=config.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        #packing=False,
        report_to="none",  # set to "wandb" if you have logged in
        remove_unused_columns=False,

    )

    print(f"Effective batch size: {config.batch_size * config.grad_accum_steps}")
    print(f"Total training steps: ~{len(data['train']) // (config.batch_size * config.grad_accum_steps) * config.num_train_epochs}")
    # --- SANITY CHECK ---

    sancheck = data['train'][0]
    text = tokenizer.apply_chat_template(
    sancheck["messages"],
    tokenize=False,
    add_generation_prompt=False,
    #enable_thinking=True ## ONLY QWEN
)


    model_inputs = tokenizer(
        text,
        add_special_tokens=True,
        return_tensors="pt"
    )

    input_ids = model_inputs["input_ids"][0].tolist()
    label = create_weighted_labels(input_ids, tokenizer, config.model_name)

    print("="*80)
    print("MASKING SANITY CHECK")
    print("="*80)

    total_tokens = len(label)
    total_unmasked = sum(1 for l in label if l != -100)
    print(f'Unmasked: {total_unmasked} of a total of {total_tokens} tokens')


    print("\n" + "="*80)
    print("INITIALIZING TRAINER")
    print("="*80)

    tokenized_train = data["train"].map(
      lambda ex: preprocess_function(ex, tokenizer, config),
      batched=True,
      remove_columns=data["train"].column_names,
      num_proc=2
      )

    tokenized_test = data["test"].map(
          lambda ex: preprocess_function(ex, tokenizer, config),
          batched=True,
          remove_columns=data["test"].column_names,
          num_proc=2
      )

    data_collator = DataCollatorForSeq2Seq(
          tokenizer=tokenizer,
          model=model,
          label_pad_token_id=-100,
          padding=True,
      )


    def validate_labels(dataset, tokenizer, num_samples=5):
        print("\n" + "="*80)
        print("VALIDATING LABELS IN DATASET")
        print("="*80)

        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            input_ids = sample['input_ids']
            labels = sample['labels']


            num_unmasked = sum(1 for l in labels if l != -100)
            num_masked = sum(1 for l in labels if l == -100)


            if num_unmasked == 0:
                print(f"WARNING: Sample {i} has ALL labels masked!")
                decoded = tokenizer.decode(input_ids[:200])
                print(f"   First 200 chars: {decoded[:200]}")
            else:
                print(f"✓ Sample {i}: {num_unmasked} unmasked, {num_masked} masked")

        print("="*80 + "\n")




    print("\nValidating training data labels...")
    validate_labels(tokenized_train, tokenizer, num_samples=5)

    print("\nValidating test data labels...")
    validate_labels(tokenized_test, tokenizer, num_samples=5)


    trainer = Trainer(
          model=model,
          args=training_args,
          train_dataset=tokenized_train,
          eval_dataset=tokenized_test,
          data_collator=data_collator,

          #processing_class=tokenizer,
      )


    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)

    # RESUME=auto  -> continue from the latest checkpoint in output_dir (e.g. after a Colab disconnect)
    # RESUME=<path> -> continue from that specific checkpoint folder
    _resume = os.environ.get("RESUME", "").strip()
    if _resume.lower() == "auto":
        from transformers.trainer_utils import get_last_checkpoint
        _resume = get_last_checkpoint(config.output_dir) if os.path.isdir(config.output_dir) else None
        print(f"Resuming from: {_resume}" if _resume else "RESUME=auto but no checkpoint found: starting from scratch")
    elif _resume:
        print(f"Resuming from: {_resume}")
    trainer.train(resume_from_checkpoint=_resume or None)




    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    print(f"✓ Model and tokenizer saved to {config.output_dir}")
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    config = FineTuneConfig()

    print("="*80)
    print("FINE-TUNING CONFIGURATION")
    print("="*80)
    print(f"Model: {config.model_name}")
    print(f"Train data: {config.train_path}")
    print(f"Test data: {config.test_path}")
    print(f"Output: {config.output_dir}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.grad_accum_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"LoRA r: {config.lora_r}, alpha: {config.lora_alpha}")
    print("="*80)

    if not os.path.exists(config.train_path):
        raise FileNotFoundError(f"Training file not found: {config.train_path}")
    if not os.path.exists(config.test_path):
        raise FileNotFoundError(f"Test file not found: {config.test_path}")

    train_model(config)