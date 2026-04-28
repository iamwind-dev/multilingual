"""Streamlit demo for comparing four local sentiment analysis models."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

# Streamlit's file watcher can inspect optional Transformers vision modules and
# print noisy torchvision errors. This text-only demo does not need hot reload.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


APP_DIR = Path(__file__).resolve().parent
LABELS = {0: "Negative", 1: "Positive"}
WEIGHT_FILES = ["model.safetensors", "pytorch_model.bin", "tf_model.h5"]
TOKENIZER_FILES = ["tokenizer.json", "vocab.txt", "sentencepiece.bpe.model", "spiece.model", "merges.txt"]
RECOMMENDED_METADATA_FILES = ["tokenizer_config.json", "special_tokens_map.json"]

MODEL_CONFIGS: list[dict[str, str]] = [
    {
        "name": "Member 1 - BERT baseline",
        "description": "English-only baseline, fine-tuned on IMDb.",
        "path": "models/member_01_bert_baseline/bert_imdb_model",
    },
    {
        "name": "Member 2 - mBERT",
        "description": "Multilingual BERT, fine-tuned on IMDb.",
        "path": "models/member_02_mbert/mbert_imdb_model",
    },
    {
        "name": "Member 3 - XLM-R",
        "description": "Cross-lingual multilingual model, fine-tuned on IMDb.",
        "path": "models/member_03_xlmr/xlmr_imdb_model",
    },
    {
        "name": "Member 4 - XLM-R Adapted",
        "description": "XLM-R fine-tuned on IMDb and adapted to Amazon multilingual reviews.",
        "path": "models/member_04_xlmr_adapted/xlmr_amazon_adapted_model",
    },
]

MODEL_OPTIONS = [model["name"] for model in MODEL_CONFIGS] + ["Compare all models"]

EXAMPLE_REVIEWS = {
    "Custom input": "",
    "English positive": "This product is amazing and works perfectly.",
    "English negative": "This product is terrible and broke after one day.",
    "Vietnamese positive": "Sản phẩm này rất tốt, giao hàng nhanh và tôi rất hài lòng.",
    "Vietnamese negative": "Sản phẩm quá tệ, pin yếu và dịch vụ hỗ trợ không tốt.",
    "French negative": "Ce produit est mauvais et ne fonctionne pas correctement.",
    "Spanish positive": "Este producto es excelente y funciona muy bien.",
    "German negative": "Das Produkt ist schlecht und sehr enttäuschend.",
    "Mixed sentiment": "The camera is excellent, but the battery life is terrible.",
}


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_absolute_model_path(model_path: str) -> Path:
    """Resolve a model path relative to the Streamlit app folder."""
    return APP_DIR / model_path


def model_status_table() -> pd.DataFrame:
    """Create a status table showing whether each configured model folder exists."""
    rows = []
    for model_config in MODEL_CONFIGS:
        absolute_path = get_absolute_model_path(model_config["path"])
        has_config = (absolute_path / "config.json").exists()
        has_weights = any((absolute_path / filename).exists() for filename in WEIGHT_FILES)
        has_tokenizer_asset = any((absolute_path / filename).exists() for filename in TOKENIZER_FILES)
        missing_required = get_missing_model_files(absolute_path)
        rows.append(
            {
                "model": model_config["name"],
                "path": model_config["path"],
                "exists": absolute_path.exists(),
                "config": has_config,
                "weights": has_weights,
                "tokenizer": has_tokenizer_asset,
                "ready": not missing_required,
                "missing_required": ", ".join(missing_required),
                "missing_recommended": ", ".join(get_missing_recommended_files(absolute_path)),
            }
        )
    return pd.DataFrame(rows)


def get_missing_model_files(model_path: Path) -> list[str]:
    """Return missing required file groups for a local Hugging Face model folder."""
    missing = []
    if not model_path.exists():
        return ["folder"]
    if not (model_path / "config.json").exists():
        missing.append("config.json")
    if not any((model_path / filename).exists() for filename in WEIGHT_FILES):
        missing.append("model.safetensors or pytorch_model.bin")
    if not any((model_path / filename).exists() for filename in TOKENIZER_FILES):
        missing.append("tokenizer.json, vocab.txt, sentencepiece.bpe.model, or spiece.model")
    return missing


def get_missing_recommended_files(model_path: Path) -> list[str]:
    """Return recommended metadata files that improve local tokenizer loading."""
    if not model_path.exists():
        return RECOMMENDED_METADATA_FILES
    return [filename for filename in RECOMMENDED_METADATA_FILES if not (model_path / filename).exists()]


def is_package_installed(package_name: str) -> bool:
    """Return whether an optional runtime package is installed."""
    return importlib.util.find_spec(package_name) is not None


def format_model_load_error(error: Exception) -> str:
    """Return an actionable model loading error message for Streamlit."""
    message = str(error)
    lowered = message.lower()
    if "sentencepiece" in lowered or "tiktoken" in lowered:
        return (
            "Tokenizer backend is missing or tokenizer files are incomplete. "
            "Run `pip install -r requirements.txt`, then restart Streamlit. "
            "For XLM-R models, the model folder must include the SentencePiece tokenizer file "
            "such as `sentencepiece.bpe.model`."
        )
    if "not found" in lowered or "no such file" in lowered or "does not appear to have" in lowered:
        return (
            "The model folder is incomplete. Copy the full Hugging Face export folder, "
            "including config, weights, tokenizer files, tokenizer_config.json, and special_tokens_map.json."
        )
    return message


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> tuple[Any, Any]:
    """Load a local Hugging Face tokenizer and sequence classification model."""
    path = str(Path(model_path))
    try:
        # Prefer slow local tokenizers to avoid fast-tokenizer conversion errors
        # for multilingual SentencePiece models such as XLM-R.
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
    return tokenizer, model


def predict_sentiment(text: str, model_name: str, model_path: str, max_length: int) -> dict[str, Any]:
    """Run sentiment prediction with one local model and return display-ready values."""
    absolute_path = get_absolute_model_path(model_path)
    result: dict[str, Any] = {
        "model": model_name,
        "prediction": None,
        "predicted_class_id": None,
        "confidence": None,
        "negative_prob": None,
        "positive_prob": None,
        "model_path": model_path,
        "status": "Not run",
    }

    missing_required = get_missing_model_files(absolute_path)
    if missing_required:
        result["status"] = "Incomplete model folder. Missing: " + "; ".join(missing_required)
        return result

    try:
        tokenizer, model = load_model(str(absolute_path))
        device = get_device()
        model.to(device)
        model.eval()

        encoded = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).squeeze(0)
            predicted_class_id = int(torch.argmax(probabilities).item())

        negative_prob = float(probabilities[0].item())
        positive_prob = float(probabilities[1].item())
        result.update(
            {
                "prediction": LABELS.get(predicted_class_id, str(predicted_class_id)),
                "predicted_class_id": predicted_class_id,
                "confidence": max(negative_prob, positive_prob),
                "negative_prob": negative_prob,
                "positive_prob": positive_prob,
                "status": "Available",
            }
        )
    except Exception as exc:
        result["status"] = f"Load or prediction failed: {format_model_load_error(exc)}"

    return result


def round_prediction_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert prediction dictionaries into a rounded table for display."""
    table = pd.DataFrame(rows)
    for column in ["confidence", "negative_prob", "positive_prob"]:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce").round(4)
    return table


def prepare_result_df(result_df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """Validate required columns and coerce probability columns to numeric."""
    missing_columns = [column for column in required_columns if column not in result_df.columns]
    if missing_columns:
        st.error("Missing required columns: " + ", ".join(missing_columns))
        return pd.DataFrame()

    prepared_df = result_df.copy()
    for column in ["confidence", "negative_prob", "positive_prob"]:
        if column in prepared_df.columns:
            prepared_df[column] = pd.to_numeric(prepared_df[column], errors="coerce")
    return prepared_df


def get_available_results(result_df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows with successful predictions."""
    if result_df.empty:
        return result_df
    if "status" not in result_df.columns:
        return result_df
    return result_df[result_df["status"] == "Available"].copy()


def plot_probability_grouped_bar(result_df: pd.DataFrame) -> None:
    """Compare negative and positive probability for each model."""
    chart_df = prepare_result_df(result_df, ["model", "negative_prob", "positive_prob"])
    chart_df = chart_df.dropna(subset=["negative_prob", "positive_prob"])
    if chart_df.empty:
        st.warning("No probability values available for charting.")
        return

    x_positions = list(range(len(chart_df)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([x - width / 2 for x in x_positions], chart_df["negative_prob"], width, label="Negative")
    ax.bar([x + width / 2 for x in x_positions], chart_df["positive_prob"], width, label="Positive")
    ax.set_title("Negative vs Positive Probability by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(chart_df["model"], rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_confidence_bar(result_df: pd.DataFrame) -> None:
    """Compare prediction confidence for each model."""
    chart_df = prepare_result_df(result_df, ["model", "confidence"])
    chart_df = chart_df.dropna(subset=["confidence"])
    if chart_df.empty:
        st.warning("No confidence values available for charting.")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(chart_df["model"], chart_df["confidence"])
    ax.set_title("Prediction Confidence by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelrotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_sentiment_vote_chart(result_df: pd.DataFrame) -> None:
    """Show how many available models predicted each sentiment label."""
    chart_df = prepare_result_df(result_df, ["prediction"])
    chart_df = chart_df.dropna(subset=["prediction"])
    if chart_df.empty:
        st.warning("No prediction labels available for vote chart.")
        return

    vote_counts = chart_df["prediction"].value_counts().reindex(["Negative", "Positive"], fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(vote_counts.index, vote_counts.values)
    ax.set_title("Sentiment Vote Across Models")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of models")
    ax.set_ylim(0, max(1, int(vote_counts.max())) + 1)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_positive_ranking_chart(result_df: pd.DataFrame) -> None:
    """Rank models by positive probability."""
    chart_df = prepare_result_df(result_df, ["model", "positive_prob"])
    chart_df = chart_df.dropna(subset=["positive_prob"]).sort_values("positive_prob", ascending=False)
    if chart_df.empty:
        st.warning("No positive probabilities available for ranking.")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(chart_df["model"], chart_df["positive_prob"])
    ax.set_title("Positive Probability Ranking")
    ax.set_xlabel("Model")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelrotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_negative_ranking_chart(result_df: pd.DataFrame) -> None:
    """Rank models by negative probability."""
    chart_df = prepare_result_df(result_df, ["model", "negative_prob"])
    chart_df = chart_df.dropna(subset=["negative_prob"]).sort_values("negative_prob", ascending=False)
    if chart_df.empty:
        st.warning("No negative probabilities available for ranking.")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(chart_df["model"], chart_df["negative_prob"])
    ax.set_title("Negative Probability Ranking")
    ax.set_xlabel("Model")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelrotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def get_majority_sentiment(result_df: pd.DataFrame) -> str:
    """Return the majority sentiment label among available predictions."""
    prepared_df = prepare_result_df(result_df, ["prediction"])
    predictions = prepared_df["prediction"].dropna()
    if predictions.empty:
        return "No prediction"
    vote_counts = predictions.value_counts()
    if len(vote_counts) > 1 and vote_counts.iloc[0] == vote_counts.iloc[1]:
        return "Tie"
    return str(vote_counts.idxmax())


def show_model_agreement(result_df: pd.DataFrame) -> None:
    """Show agreement or disagreement across available model predictions."""
    prepared_df = prepare_result_df(result_df, ["prediction"])
    predictions = prepared_df["prediction"].dropna()
    if predictions.empty:
        st.warning("No available model predictions for agreement analysis.")
        return

    positive_votes = int((predictions == "Positive").sum())
    negative_votes = int((predictions == "Negative").sum())
    st.write(f"Positive votes: {positive_votes}")
    st.write(f"Negative votes: {negative_votes}")

    if predictions.nunique() == 1:
        st.success("All available models agree on the sentiment.")
    else:
        st.warning("Models disagree. This may indicate domain shift, language shift, or mixed sentiment.")


def show_summary_metrics(result_df: pd.DataFrame) -> None:
    """Show quick comparison metrics above result tables."""
    prepared_df = prepare_result_df(result_df, ["model", "prediction", "confidence"])
    prepared_df = prepared_df.dropna(subset=["prediction", "confidence"])
    if prepared_df.empty:
        st.warning("No available model results for summary metrics.")
        return

    highest_confidence_row = prepared_df.sort_values("confidence", ascending=False).iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Available models", len(prepared_df))
    with col2:
        st.metric("Highest confidence model", str(highest_confidence_row["model"]))
    with col3:
        st.metric("Highest confidence score", f"{highest_confidence_row['confidence']:.4f}")
    with col4:
        st.metric("Final majority sentiment", get_majority_sentiment(prepared_df))


def render_chart_tabs(result_df: pd.DataFrame, include_votes: bool) -> None:
    """Render probability, vote, ranking, and business-insight tabs."""
    available_df = get_available_results(result_df)
    if available_df.empty:
        st.warning("No successful prediction results available for charts.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Probability", "Votes", "Ranking", "Business Insight"])
    with tab1:
        plot_probability_grouped_bar(available_df)
        plot_confidence_bar(available_df)
    with tab2:
        if include_votes or len(available_df) > 1:
            plot_sentiment_vote_chart(available_df)
        show_model_agreement(available_df)
    with tab3:
        plot_positive_ranking_chart(available_df)
        plot_negative_ranking_chart(available_df)
    with tab4:
        if len(available_df) == 1:
            show_single_interpretation(available_df.iloc[0].to_dict())
        else:
            show_comparison_interpretation(available_df)
        st.write("High positive probability means stronger customer satisfaction signal.")
        st.write("High negative probability means stronger customer dissatisfaction signal.")
        st.write("Model disagreement can happen with multilingual text, domain shift, or mixed sentiment.")


def show_single_interpretation(result: dict[str, Any]) -> None:
    """Display a business interpretation for a single model result."""
    prediction = result.get("prediction")
    if prediction == "Negative":
        st.warning("Business action: This review may require attention from customer support.")
    elif prediction == "Positive":
        st.success("Business action: This review indicates customer satisfaction.")


def show_comparison_interpretation(results_df: pd.DataFrame) -> None:
    """Display automatic interpretation for all-model comparison."""
    available = results_df[results_df["status"] == "Available"].copy()
    if available.empty:
        st.warning("No available model produced a prediction.")
        return

    highest_confidence_row = available.sort_values("confidence", ascending=False).iloc[0]
    predictions = set(available["prediction"].dropna().tolist())
    st.info(
        f"Highest confidence: {highest_confidence_row['model']} "
        f"({highest_confidence_row['confidence']:.4f})."
    )
    if len(predictions) == 1:
        prediction = next(iter(predictions))
        st.success(f"All available models agree on {prediction}.")
    else:
        st.warning("Models disagree on the sentiment prediction.")
        st.write("Model disagreement may indicate domain shift, language shift, or mixed sentiment.")


def get_selected_text(example_name: str) -> str:
    """Return the example text for the selected example name."""
    return EXAMPLE_REVIEWS.get(example_name, "")


def render_sidebar() -> tuple[str, int]:
    """Render sidebar controls and return selected model option plus max length."""
    st.sidebar.header("Controls")
    selected_model = st.sidebar.selectbox("Model selection", MODEL_OPTIONS)
    max_length = st.sidebar.slider("Max length", min_value=64, max_value=512, value=256, step=64)

    device = get_device()
    st.sidebar.subheader("Device info")
    if device.type == "cuda":
        st.sidebar.write(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        st.sidebar.write("Using CPU")

    st.sidebar.subheader("Model status checker")
    st.sidebar.dataframe(model_status_table(), width="stretch", hide_index=True)
    st.sidebar.caption(
        "sentencepiece: "
        + ("installed" if is_package_installed("sentencepiece") else "missing - run pip install -r requirements.txt")
    )
    return selected_model, max_length


def render_model_explanations() -> None:
    """Show a short explanation of the four compared models."""
    st.header("Model Comparison Explanation")
    for model_config in MODEL_CONFIGS:
        st.markdown(f"**{model_config['name']}**: {model_config['description']}")


def main() -> None:
    st.set_page_config(page_title="Multilingual Sentiment Analysis Demo", layout="wide")

    st.title("Multilingual Sentiment Analysis Demo")
    st.subheader("IMDb to Amazon Cross-Domain and Cross-Lingual Transfer Learning")
    st.write(
        "This local demo compares four exported Hugging Face sentiment models from the group project. "
        "Enter a customer review, choose one model or compare all models, and inspect sentiment probabilities."
    )

    selected_model, max_length = render_sidebar()

    st.header("Review Input")
    example_name = st.selectbox("Example review selector", list(EXAMPLE_REVIEWS.keys()))
    default_text = get_selected_text(example_name)
    review_text = st.text_area("Customer review", value=default_text, height=140)
    analyze = st.button("Analyze", type="primary")

    if analyze:
        if not review_text.strip():
            st.warning("Please enter a review before analyzing.")
        elif selected_model == "Compare all models":
            rows = []
            for model_config in MODEL_CONFIGS:
                result = predict_sentiment(
                    review_text.strip(),
                    model_config["name"],
                    model_config["path"],
                    max_length,
                )
                if result["status"] != "Available":
                    st.warning(f"{model_config['name']}: {result['status']}")
                rows.append(result)

            results_df = round_prediction_rows(rows)
            st.subheader("Comparison Results")
            available_df = get_available_results(results_df)
            if not available_df.empty:
                show_summary_metrics(available_df)
            st.dataframe(
                results_df[["model", "prediction", "confidence", "negative_prob", "positive_prob", "status"]],
                width="stretch",
                hide_index=True,
            )

            st.subheader("Charts")
            render_chart_tabs(results_df, include_votes=True)
        else:
            model_config = next(model for model in MODEL_CONFIGS if model["name"] == selected_model)
            result = predict_sentiment(review_text.strip(), model_config["name"], model_config["path"], max_length)
            if result["status"] != "Available":
                st.error(result["status"])
            else:
                single_result_df = round_prediction_rows([result])
                st.subheader("Prediction Result")
                show_summary_metrics(single_result_df)
                st.dataframe(
                    single_result_df[
                        [
                            "model",
                            "prediction",
                            "predicted_class_id",
                            "confidence",
                            "negative_prob",
                            "positive_prob",
                            "model_path",
                            "status",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )
                st.subheader("Charts")
                render_chart_tabs(single_result_df, include_votes=False)

    render_model_explanations()


if __name__ == "__main__":
    main()
