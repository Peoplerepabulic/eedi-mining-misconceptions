# Eedi: Mining Misconceptions in Mathematics 🥈

Silver Medal solution (rank **42 / 1,446** teams) for the
[Kaggle competition](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics)
on predicting affinities between math multiple-choice distractors and underlying
student misconceptions, evaluated with **MAP@25**.

## Problem

Manually matching each wrong answer (distractor) to the misconception behind it
is slow and inconsistent. This project automates that matching with NLP:
given a question's distractors, predict the most likely misconceptions —
covering known ones and generalizing to new ones.

## Approach

**1. Data generation & training**
- Used GPT-4o to generate synthetic training data for misconceptions missing
  from the official dataset.
- Fine-tuned **Qwen2.5-14B** and **Qwen2.5-32B-Instruct** with contrastive
  learning (SimCSE-style), using 4-bit quantization and **LoRA** adapters on
  multiple linear modules (`code/qwen_v10.py`, `code/llm_n6.py`).

**2. Retrieval & ensemble**
- Concatenated embeddings from three differently-trained LLMs and retrieved
  candidates by similarity.

**3. Reranking**
- Reranked the top-25 candidates with **Qwen2.5-32B-Instruct-AWQ** (zero-shot,
  processed in batches) and **bge-large-en-v1.5** similarity search.

## Repository layout

```
code/
  llm_n6.py      # Contrastive fine-tuning script (Qwen2.5-32B, 4-bit + LoRA)
  qwen_v10.py    # Contrastive fine-tuning script (Qwen2.5-14B, 4-bit + LoRA)
  infer.ipynb    # Inference notebook
data/
  misconception_mapping.csv                 # MisconceptionId -> MisconceptionName
  val_misconceptions.csv                    # Validation split
  misconception_data_all_train_gpt-4o.xlsx  # Training data incl. GPT-4o synthetic rows
docs/
  Eedi_____4.docx  # Full solution write-up (Chinese + English)
```

## Notes

- Training scripts expect model weights and data paths as configured in the
  `VER` / `DATA_PATH_*` / `MODEL_PATH` constants — adjust to your environment.
  Full Qwen weights are not included here.
- The synthetic training rows were generated with GPT-4o to augment the
  official competition data.

## Author

Qiwei Li — Washington University in St. Louis (CS + Math, '27)
