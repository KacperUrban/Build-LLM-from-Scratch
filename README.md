# Build-LLM-from-Scratch
This repo is a part of my study on LLMs based on great book from Sebastian Raschka "Build a Large Language Model (From Scratch)". I will upload all my code and experimantions into this repository. Code written from different chapter will be placed in each folder or different notebook. I will updated repo as things progress.

# Mlflow
Run mlflow server locally for tracking training:
```
mlflow server --port 5000
```

# Run pretraining on HF dataset
To run pretraining use script: pretrain_llm_on_hf_dataset.py, where you should define target dataset and of course optionally you can log
all metrics in mlflow for further investigation.