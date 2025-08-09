"""
HRM-Text1 Training Script - based on https://github.com/qingy1337/HRM-Text

who took inspiration from [SofiTesfay2010's script](https://colab.research.google.com/drive/1xZNYC-yhwdJxzbpwRekE_rDjTki5CvEv?usp=sharing)
"""

import os
import shutil
import random
import datetime

# Import model classes
from hrm_text1_modeling import HRMText1

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import load_dataset, interleave_datasets
from itertools import chain
from tqdm.auto import tqdm

from huggingface_hub import HfApi, HfFolder
import wandb

# ----------------------------
# Training Parameters
SEED = 42
NUM_EPOCHS = 2
BLOCK_SIZE = 512
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4  # Effective batch size = 8 * 4 = 32
LEARNING_RATE_MAX = 2e-4
LEARNING_RATE_MIN = 1e-6
WEIGHT_DECAY = 0.01
MIXED_PRECISION = True
EARLY_STOPPING_PATIENCE = 2  # Stop if validation loss doesn't improve for 2 epochs

SAVE_STEPS = 500  # Save a checkpoint every 500 global steps

# HRM Model Hyperparameters
MODEL_CONFIG = {"d_model": 512, "n_heads": 8, "d_ff": 2048, "dropout": 0.1}
MAX_HALT_STEPS = 8
PONDER_WEIGHT = 1e-2
PONDER_WEIGHT_DECAY = 0.98  # Decay ponder weight each epoch to focus on LM loss later
HALT_BIAS_INIT = (
    -2.2
)  # Halt bias encourages more steps early on (sigmoid(-2.2) approx 0.1)

T5_TOKENIZER_REPO = "t5-small"
LOCAL_CHECKPOINT_PATH = "local_training_state.pt"
LOCAL_WEIGHTS_PATH = "pytorch_model.bin"
BEST_MODEL_PATH = "best_model.bin"

WANDB_PROJECT = "HRM-Text1"
UPDATE_README = True


# ----------------------------
# Utilities & Initialization
def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"Using device: {device}")

# HuggingFace & W&B Authentication
try:
    HF_TOKEN = os.environ["HF_TOKEN"]
    WANDB_API_KEY = os.environ["WANDB_API_KEY"]

    print("Hugging Face & W&B tokens loaded.")
    HfFolder.save_token(HF_TOKEN)
    wandb.login(key=WANDB_API_KEY)

    print("Login to Hugging Face Hub and W&B successful.")
except Exception as e:
    print("HF_TOKEN or WANDB_API_KEY secret not found.")
    HF_TOKEN, WANDB_API_KEY = None, None

# ----------------------------
# Tokenizer
from transformers import AutoTokenizer

# tok_name = "EleutherAI/gpt-neox-20b"
# NOTE: we use t5 tokenizer for simplicity for now
tok_name = T5_TOKENIZER_REPO
print(f"Loading tokenizer from {tok_name}...")
tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
# GPT2/NeoX-style tokenizers benefit from this for leading spaces:
tokenizer.add_prefix_space = True
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# SEQ_LEN = 2048  # bump if HRM can handle longer; PG-19 benefits from 4k–8k
# NOTE: we set it to 512 for now local testing
SEQ_LEN = BLOCK_SIZE  # Use the same block size as training


def tokenize(batch):
    return tokenizer(batch["text"], truncation=False, add_special_tokens=False)


# Efficient “concat & chunk” packer: turns a stream of token lists into fixed blocks
# def pack_blocks(batch):
#     ids = list(chain.from_iterable(batch["input_ids"]))
#     # Insert EOS between docs to avoid run-on contamination
#     eos = tokenizer.eos_token_id
#     with_eos = []
#     for seq in batch["input_ids"]:
#         with_eos.extend(seq + [eos])
#     ids = with_eos
#     blocks = []
#     for i in range(0, len(ids) - SEQ_LEN, SEQ_LEN):
#         blocks.append({"input_ids": ids[i : i + SEQ_LEN]})
#     return {"input_ids": [b["input_ids"] for b in blocks]}


def pack_sequences(examples):
    """Pack sequences into fixed-length blocks without batching issues"""
    all_ids = []
    eos_id = tokenizer.eos_token_id

    # Flatten all sequences and add EOS tokens
    for seq in examples["input_ids"]:
        all_ids.extend(seq + [eos_id])

    # Create fixed-size blocks
    blocks = []
    for i in range(0, len(all_ids) - SEQ_LEN + 1, SEQ_LEN):
        blocks.append(all_ids[i : i + SEQ_LEN])

    return {"input_ids": blocks}


def to_labels(example):
    x = example["input_ids"]
    return {"input_ids": x, "labels": x.copy(), "attention_mask": [1] * len(x)}


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
tokenizer.padding_side = "left"  # Important for causal modeling during generation
print(
    f"Tokenizer loaded. Vocab size: {len(tokenizer)}; eos={tokenizer.eos_token}; pad={tokenizer.pad_token}"
)


# -----------------------------------
# Data Loading and Preprocessing
print("Loading datasets...")
# --- TEXT SOURCES ---

# NOTE: we use tinystories here as a small dataset for quick testing.
# You can replace it with any other text dataset you prefer.
train_mixture = load_dataset("roneneldan/TinyStories", split="train")
eval_mixture = load_dataset("roneneldan/TinyStories", split="validation")


# fineweb = load_dataset(
#     "HuggingFaceFW/fineweb", name="default", split="train", streaming=True
# )  # or "base"
# slimpajama = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
# pg19 = load_dataset("deepmind/pg19", split="train", streaming=True)
# # --- CODE SOURCES ---
# code_stream = load_dataset("bigcode/starcoderdata", split="train", streaming=True)

# # --- TEXT SOURCES ---
# fineweb_eval = load_dataset(
#     "HuggingFaceFW/fineweb", name="default", split="validation", streaming=True
# )  # or "base"
# slimpajama_eval = load_dataset(
#     "cerebras/SlimPajama-627B", split="validation", streaming=True
# )
# pg19_eval = load_dataset("deepmind/pg19", split="validation", streaming=True)
# # --- CODE SOURCES ---
# code_stream_eval = load_dataset(
#     "bigcode/starcoderdata", split="validation", streaming=True
# )

# TODO: Use a more comprehensive code dataset
# Prefer The Stack v2 if you have access
# try:
#     stack = load_dataset("bigcode/the-stack-v2", split="train", streaming=True)
#     # keep only permissive licenses
#     PERMISSIVE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "mpl-2.0", "unlicense"}
#     stack = stack.filter(lambda r: (r.get("license", "") or "").lower() in PERMISSIVE)
#     code_stream = stack
# except Exception:
# fallback: StarCoderData
#     code_stream = load_dataset("bigcode/starcoderdata", split="train", streaming=True)


def get_text(example):
    # Normalize fields across datasets
    if "text" in example and isinstance(example["text"], str):
        return {"text": example["text"]}
    for k in ("content", "content_text", "body", "dump"):
        if k in example and isinstance(example[k], str):
            return {"text": example[k]}
    return {"text": ""}


# fineweb = fineweb.map(get_text)
# slimpajama = slimpajama.map(get_text)
# pg19 = pg19.map(lambda ex: {"text": ex.get("text", ex.get("book_text", ""))})
# code_stream = code_stream.map(get_text)

# Interleave with weights (FineWeb 0.3, SlimPajama 0.2, PG-19 0.2, Code 0.3)
# train_mixture = interleave_datasets(
#     [fineweb, slimpajama, pg19, code_stream],
#     probabilities=[0.3, 0.2, 0.2, 0.3],
#     seed=42,
# )

# fineweb_eval = fineweb_eval.map(get_text)
# slimpajama_eval = slimpajama_eval.map(get_text)
# pg19_eval = pg19_eval.map(lambda ex: {"text": ex.get("text", ex.get("book_text", ""))})
# code_stream_eval = code_stream_eval.map(get_text)

# eval_mixture = interleave_datasets(
#     [fineweb_eval, slimpajama_eval, pg19_eval, code_stream_eval],
#     probabilities=[0.3, 0.2, 0.2, 0.3],
#     seed=42,
# )

# Tokenize and pack into blocks
print("Tokenizing and packing blocks...")

tok_stream_train = train_mixture.map(
    tokenize, batched=True, remove_columns=train_mixture.column_names
)

# NOTE: we set the batch size to 500 for packing sequences - otherwise the current model's
# tokenization will not be able to handle large sequences and the resulting packed
tok_stream_train = tok_stream_train.map(
    pack_sequences,
    batched=True,
    remove_columns=tok_stream_train.column_names,
    batch_size=500,
)

tok_stream_eval = eval_mixture.map(
    tokenize, batched=True, remove_columns=eval_mixture.column_names
)

tok_stream_eval = tok_stream_eval.map(
    pack_sequences,
    batched=True,
    remove_columns=tok_stream_eval.column_names,
    batch_size=500,
)

print(f"Train dataset size: {len(tok_stream_train)} blocks")
print(f"Eval dataset size: {len(tok_stream_eval)} blocks")
# Convert to labels for training
# NOTE: (input_ids are the same as labels for causal language modeling)
print("Converting to labels...")

train_dataset = tok_stream_train.map(to_labels)
eval_dataset = tok_stream_eval.map(to_labels)

# Add a custom data collator to properly handle the data
from transformers import DataCollatorForLanguageModeling

# Create a data collator that handles padding and converts to tensors
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal LM, not masked LM
    pad_to_multiple_of=None,
    return_tensors="pt",
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    drop_last=True,
    collate_fn=data_collator,  # Add the collator
)

val_loader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    collate_fn=data_collator,  # Add the collator
)

# --------------------------------
# Model, Optimizer, Scheduler
config = {
    "vocab_size": len(tokenizer),
    "block_size": BLOCK_SIZE,
    "d_model": MODEL_CONFIG["d_model"],
    "n_heads": MODEL_CONFIG["n_heads"],
    "d_ff": MODEL_CONFIG["d_ff"],
    "dropout": MODEL_CONFIG["dropout"],
    "halt_max_steps": MAX_HALT_STEPS,
    "ponder_loss_weight": PONDER_WEIGHT,
    "halt_bias_init": HALT_BIAS_INIT,
}
model = HRMText1(config).to(device)

decay, no_decay = [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.dim() == 1 or any(k in n.lower() for k in ["bias", "norm"]):
        no_decay.append(p)
    else:
        decay.append(p)

optimizer = AdamW(
    [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=LEARNING_RATE_MAX,
    betas=(0.9, 0.95),
    eps=1e-8,
)

# TODO: Load pre-trained weights if available
# try:
#     print(f"Downloading latest model from '{HF_REPO_ID}'...")
#     weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=LOCAL_WEIGHTS_PATH)
#     state = torch.load(weights_path, map_location=device)
#     model.load_state_dict(state, strict=False)
#     print("Successfully loaded model from the Hub. Continuing training.")
# except Exception as e:
#     print(f"Could not download model from Hub. Starting fresh. Error: {e}")

start_epoch, global_step = 0, 0
if os.path.exists(LOCAL_CHECKPOINT_PATH):
    try:
        chk = torch.load(LOCAL_CHECKPOINT_PATH, map_location="cpu")
        optimizer.load_state_dict(chk["optimizer_state_dict"])
        start_epoch = chk.get("epoch", -1) + 1
        global_step = chk.get("global_step", 0)
        print(f"Resuming from Epoch {start_epoch}, global_step {global_step}.")
    except Exception as e:
        print(f"Warning: failed to load local training state: {e}")

steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
# NOTE: this assumes that the dataset loader does not stream the dataset
# if it does, we will need to calculate the number of steps differently, i.e.
# with this estimation:
# ESTIMATED_SAMPLES_PER_EPOCH = 10000  # Adjust based on your expected dataset size
# steps_per_epoch = ESTIMATED_SAMPLES_PER_EPOCH // (BATCH_SIZE * GRAD_ACCUM_STEPS)

num_training_steps = NUM_EPOCHS * steps_per_epoch
scheduler = CosineAnnealingLR(
    optimizer, T_max=num_training_steps, eta_min=LEARNING_RATE_MIN
)

scaler = torch.cuda.amp.GradScaler(enabled=(MIXED_PRECISION and device.type == "cuda"))

# ----------------------------
# Training Loop
wandb.init(
    project=WANDB_PROJECT,
    config=config,
    name=f"run-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}",
)
wandb.watch(model, log="all", log_freq=steps_per_epoch)

best_val_loss = float("inf")
patience_counter = 0

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    # Update ponder weight for this epoch
    current_ponder_weight = PONDER_WEIGHT * (PONDER_WEIGHT_DECAY**epoch)
    model.ponder_loss_weight = current_ponder_weight

    progress = tqdm(
        train_loader, desc=f"Epoch {epoch} | Ponder Weight: {current_ponder_weight:.4f}"
    )
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(progress):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(
            device
        )  # Use the labels from batch instead of input_ids

        with torch.amp.autocast(
            "cuda", enabled=(MIXED_PRECISION and device.type == "cuda")
        ):
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs["loss"]
            lm_loss = outputs.get("lm_loss", torch.tensor(0.0))
            ponder_loss = outputs.get("ponder_loss", torch.tensor(0.0))

        if loss is None or not torch.isfinite(loss):
            print("Non-finite loss, skipping batch.")
            optimizer.zero_grad(set_to_none=True)
            continue

        loss_to_backprop = loss / GRAD_ACCUM_STEPS
        scaler.scale(loss_to_backprop).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            wandb.log(
                {
                    "train/step_loss": loss.item(),
                    "train/lm_loss": lm_loss.item(),
                    "train/ponder_loss": ponder_loss.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item(),
                    "global_step": global_step,
                }
            )

            if global_step > 0 and global_step % SAVE_STEPS == 0:
                print(f"\nSaving checkpoint at global_step {global_step}")
                # Create a dedicated directory for this checkpoint
                checkpoint_dir = f"checkpoint-{global_step}"
                os.makedirs(checkpoint_dir, exist_ok=True)

                # Save model weights and full training state for resuming
                model_path = os.path.join(checkpoint_dir, LOCAL_WEIGHTS_PATH)
                state_path = os.path.join(checkpoint_dir, LOCAL_CHECKPOINT_PATH)

                torch.save(model.state_dict(), model_path)
                torch.save(
                    {
                        "epoch": epoch,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                    },
                    state_path,
                )

                # TODO: Upload the checkpoint folder to Hugging Face Hub
                # try:
                #     api = HfApi()
                #     commit_msg = f"Checkpoint at step {global_step} (Epoch {epoch})"
                #     api.upload_folder(
                #         folder_path=checkpoint_dir,
                #         repo_id=HF_REPO_ID,
                #         repo_type="model",
                #         commit_message=commit_msg,
                #     )
                #     print(
                #         f"Successfully uploaded checkpoint {global_step} to {HF_REPO_ID}"
                #     )
                #     # Clean up the local folder after a successful upload
                #     shutil.rmtree(checkpoint_dir)
                # except Exception as e:
                #     print(f"Upload failed for checkpoint {global_step}: {e}")

        progress.set_postfix(
            {"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"}
        )

    # Validation
    model.eval()
    total_val_loss = 0.0
    with torch.inference_mode():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)  # Use labels from batch
            out = model(input_ids, labels=labels, attention_mask=attention_mask)
            if out["loss"] is not None and torch.isfinite(out["loss"]):
                total_val_loss += out["loss"].item()

    avg_val_loss = total_val_loss / max(1, len(val_loader))
    val_perplexity = torch.exp(torch.tensor(avg_val_loss))
    print(
        f"\nEpoch {epoch} | Validation Loss: {avg_val_loss:.4f} | Validation Perplexity: {val_perplexity:.2f}"
    )
    wandb.log(
        {"epoch": epoch, "val/loss": avg_val_loss, "val/perplexity": val_perplexity}
    )

    # Save Model & Check for Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        print(
            f"New best validation loss: {best_val_loss:.4f}. Saving best model to '{BEST_MODEL_PATH}'"
        )
        torch.save(model.state_dict(), BEST_MODEL_PATH)
    else:
        patience_counter += 1
        print(
            f"Validation loss did not improve. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
        )

    # Save full checkpoint for resuming (this one is for local recovery)
    torch.save(
        {
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        },
        LOCAL_CHECKPOINT_PATH,
    )

    # TODO: upload Model to Hugging Face Hub?
    # print("\nUploading best model from epoch to HuggingFace Hub...")
    # try:
    #     api = HfApi()
    #     # Upload the best model to the root of the repo
    #     api.upload_file(
    #         path_or_fileobj=BEST_MODEL_PATH,
    #         path_in_repo=LOCAL_WEIGHTS_PATH,  # Overwrites the main model file
    #         repo_id=HF_REPO_ID,
    #         repo_type="model",
    #         commit_message=f"End of Epoch {epoch}: Val Loss {avg_val_loss:.4f}, Perplexity {val_perplexity:.2f}",
    #         token=HF_TOKEN,
    #     )

    #     print("Finished upload to repo!")
    # except Exception as e:
    #     print(f"Upload failed: {e}")

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

wandb.finish()
print("Training run finished.")


# ----------------------------
# Chatting!
def chat_with_model(
    prompt_text, model, tokenizer, max_new_tokens=60, temperature=0.7, top_k=50
):
    model.eval()
    input_ids = tokenizer.encode(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            out = model(input_ids, attention_mask=attention_mask)
            next_token_logits = out["logits"][:, -1, :] / max(temperature, 1e-6)

            # Top-K filtering
            topk_vals, topk_idx = torch.topk(
                next_token_logits, k=min(top_k, next_token_logits.size(-1))
            )
            mask = torch.full_like(next_token_logits, float("-inf"))
            mask.scatter_(1, topk_idx, topk_vals)

            probs = F.softmax(mask, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=device, dtype=torch.long)],
                dim=1,
            )

            if (
                tokenizer.eos_token_id is not None
                and next_token_id.item() == tokenizer.eos_token_id
            ):
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


print("\nSampling Generation...")

if os.path.exists(BEST_MODEL_PATH):
    print("Loading best model for generation...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    try:
        print(
            chat_with_model("Hello, how are you?", model, tokenizer, max_new_tokens=30)
        )
        print(
            chat_with_model(
                "The meaning of life is", model, tokenizer, max_new_tokens=50
            )
        )
    except Exception as e:
        print(f"Generation test failed: {e}")
else:
    print("Best model file not found. Could not run generation test.")
