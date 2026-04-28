# Multilingual Transfer Learning for Cross-Domain Sentiment Analysis: IMDb to Amazon Reviews

## Purpose

This Streamlit web demo compares four exported Hugging Face sentiment analysis models from the group project. Users can enter a customer review, choose one model or compare all models, and view predicted sentiment, confidence, probabilities, charts, and a short business interpretation.

## Folder Structure

```text
sentiment_web_demo/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   ├── member_01_bert_baseline/
│   │   └── bert_imdb_model/
│   ├── member_02_mbert/
│   │   └── mbert_imdb_model/
│   ├── member_03_xlmr/
│   │   └── xlmr_imdb_model/
│   └── member_04_xlmr_adapted/
│       └── xlmr_amazon_adapted_model/
└── sample_inputs/
    └── sample_reviews.csv
```

## Where To Put The Exported Models

Place each complete local Hugging Face model folder here:

- Member 1 - BERT baseline: `models/member_01_bert_baseline/bert_imdb_model`
- Member 2 - mBERT: `models/member_02_mbert/mbert_imdb_model`
- Member 3 - XLM-R: `models/member_03_xlmr/xlmr_imdb_model`
- Member 4 - XLM-R Adapted: `models/member_04_xlmr_adapted/xlmr_amazon_adapted_model`

Each folder should include files such as `config.json`, `model.safetensors` or `pytorch_model.bin`, tokenizer files, `tokenizer_config.json`, and `special_tokens_map.json`.

## Install Dependencies

```bash
pip install -r requirements.txt
```

If Streamlit was already running before installing dependencies, stop it and run it again so Python can load newly installed tokenizer packages such as `sentencepiece`.

## Run The App

```bash
streamlit run app.py
```

## Notes

Do not upload only `model.safetensors`. Each model must be a complete Hugging Face model folder with model configuration and tokenizer files.
