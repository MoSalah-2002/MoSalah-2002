# Arabic Fatawa Question Answering — LoRA Fine-Tuning

An experimental supervised fine-tuning project for adapting the pretrained Qwen3-4B causal language model to Arabic question-answer pairs from a fatawa dataset.

## Approach
- Clean and deduplicate question-answer records.
- Filter very short answers.
- Convert each pair into a user/assistant conversation.
- Apply parameter-efficient fine-tuning with LoRA.
- Train with TRL's SFTTrainer and evaluate on a held-out split.

## Configuration
- Base model: `Qwen/Qwen3-4B`
- LoRA rank: 64; alpha: 128; dropout: 0.1
- Target modules: q_proj, k_proj, v_proj
- Training subset: up to 10,000 records
- Split: 70% train / 30% evaluation
- Epochs: 1

## Dataset
The original notebook references the Kaggle dataset `abdallahelsaadany/fatawa`. Download it from Kaggle and place the CSV at `data/88kData.csv`. The dataset is not included in this repository.

## Run
Install dependencies from `requirements.txt`, prepare the dataset, then run `notebooks/fakeeh-chatbot.ipynb` in a GPU-enabled environment. Qwen3-4B fine-tuning requires substantial GPU memory; adjust batch size and use gradient checkpointing/quantization if needed.

## Limitations
This is an experimental model adaptation, not a validated religious authority or a production fatwa service. Generated answers must be checked against reliable scholarly sources. The notebook's training setup and evaluation should be improved before making quality claims; no benchmark score is claimed.

## Stack
Python, PyTorch, Hugging Face Transformers, Datasets, PEFT, TRL.
