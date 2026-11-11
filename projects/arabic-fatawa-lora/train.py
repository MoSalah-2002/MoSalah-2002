"""Experimental LoRA fine-tuning of Qwen3-4B on Arabic fatawa QA pairs.

Prepare data/88kData.csv with columns: ID, ques, ans, title.
Run only on a GPU runtime with enough memory.
"""
import re
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen3-4B"
DATA_PATH = "data/88kData.csv"

def clean_text(text: str) -> str:
    # Preserve Arabic letters, Latin letters, whitespace and digits.
    text = re.sub(r"[^\w\s]", " ", str(text), flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

def main():
    df = pd.read_csv(DATA_PATH).drop(columns=["ID"], errors="ignore")
    df = df.dropna(subset=["ques", "ans"]).drop_duplicates()
    df["ques"] = df["ques"].map(clean_text)
    df["ans"] = df["ans"].map(clean_text)
    df = df[df["ans"].str.len() >= 118].reset_index(drop=True)
    df = df.head(10_000)
    if len(df) < 2:
        raise ValueError("Not enough usable question-answer rows.")

    ds = Dataset.from_pandas(df[["ques", "ans"]], preserve_index=False)
    ds = ds.train_test_split(test_size=0.3, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map="auto"
    )

    def to_conversation(row):
        return {"messages": [
            {"role": "user", "content": row["ques"]},
            {"role": "assistant", "content": row["ans"]},
        ]}

    train_ds = ds["train"].map(to_conversation, remove_columns=ds["train"].column_names)
    eval_ds = ds["test"].map(to_conversation, remove_columns=ds["test"].column_names)

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False,
        r=64, lora_alpha=128, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj"], bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = SFTConfig(
        output_dir="outputs/qwen3-fatawa-lora",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=train_ds,
        eval_dataset=eval_ds, processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model("outputs/qwen3-fatawa-lora/final-adapter")
    tokenizer.save_pretrained("outputs/qwen3-fatawa-lora/final-adapter")

if __name__ == "__main__":
    main()
