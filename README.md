# Text Normalization with LLMs

An LLM-based text normalization system that uses OpenRouter API to normalize Russian text sentences. The project evaluates various LLM models on their ability to normalize text according to linguistic rules.

The code is prepared for the article «Assessing the Applicability of Frontier LLMs for Russian Social Media Text Normalization» (2026)

## Environment Setup

### 1. Install Python Dependencies

Using `pyproject.toml`:
```bash
pip install .
```

Or using `requirements.txt` for pinned versions:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and add your OpenRouter API key:
```bash
cp .env.example .env
```

Edit `.env` and fill in:
```
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_API_URL=https://openrouter.ai/api/v1
CONTENT_TYPE=application/json
```

Get your API key from [OpenRouter](https://openrouter.ai).

## Quickstart

### Basic Usage

Run the evaluation with the default configuration:
```bash
python3 main.py
```

### Configure Model and Parameters

Edit `configs/config.yaml` to change:
- `model`: LLM model name (e.g., `anthropic/claude-3.5-sonnet`, `openai/gpt-4o`)
- `batch_size`: Number of sentences per request (1-7 recommended)
- `prompt`: Which prompt template to use from `configs/prompts/` (should be written without .yaml extensions, e.g. `prompt=protocol_based`)
- `reasoning`: Enable model reasoning (`minimal`, `low`, `medium`, `high`, or `null`)  (beware that max tokens limit can be set in `main.py -> send_request(...)`)
- `test_size`: Train/test split ratio (0.0-1.0)

Do not change:
- `num_sentences`: Limit number of sentences from original dataset (should be set to 2000 by default)

Example:
```bash
python3 main.py model=anthropic/claude-opus-4-6 batch_size=4 prompt=instructions_based
```

Beware that Anthopic models from Claude 4.6 family work only with adaptive thinking: https://openrouter.ai/docs/guides/evaluate-and-optimize/model-migrations/claude-4-6

## Project Structure

```
normalization-llm/
├── main.py                 # Main evaluation loop
├── dataset.py              # Dataset loading and preprocessing
├── metrics.py              # Evaluation metrics computation
├── config.py               # Environment configuration
├── pyproject.toml          # Project dependencies (pip install .)
├── requirements.txt        # Pinned dependency versions
├── .env.example            # Template for environment variables
├── .env                    # Local environment variables (git-ignored)
│
├── configs/                # Hydra configuration files
│   ├── config.yaml         # Main configuration
│   ├── prompts/            # LLM prompt templates
│       └── *.yaml          # Different prompt variants
│
├── dataset.csv             # Dataset with original and normalized sentences
├── predictions/            # Generated prediction CSVs
├── outputs/                # Hydra outputs (logs, configs)
└── wandb/                  # Weights & Biases experiment tracking
```
