import requests
import re
import csv
import time
import logging
import signal
import sys

import wandb
import yaml

from pathlib import Path
from typing import Tuple, List
from math import ceil
from tqdm import tqdm

from dataset import get_train_test, PROMPT_SENT_EXAMPLES
from metrics import compute_metrics
from config import settings

from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf


logging.disable(logging.INFO)
signal.signal(signal.SIGINT, signal.SIG_DFL)


def load_prompt(system_prompt: str, batch_size: int) -> str:
    """Load a system prompt from the prompts directory 
    and augment it with example sentences

    Args:
        prompt: Name of the prompt file (without .yaml extension)
        batch_size: Number of sentences given as input to the model (for number of examples)

    Returns:
        prompt (str): System prompt text
    """
    prompts_path = Path(__file__).parent / "configs" / "prompts"
    with open(prompts_path / f"{system_prompt}.yaml") as f:
        data = yaml.safe_load(f)
        
    system_prompt = augment_prompt_with_examples(data["prompt"], batch_size)
        
    return system_prompt


def augment_prompt_with_examples(prompt: str, batch_size: int) -> str:
    """Add example sentences to the prompt (matching batch_size)

    Args:
        prompt (str): System prompt
        batch_size (int): Number of sentences in the batch

    Returns:
        str: System prompt with examples in the end
    """
    num_sents = len(PROMPT_SENT_EXAMPLES["original"])
    assert batch_size <= num_sents
    
    examples = "\n\n"
    for i in range(num_sents//batch_size):
        start = i*batch_size
        end = (i+1)*batch_size
        
        examples += "<example_input>\n"
        examples += "<S> " + " <S> ".join(PROMPT_SENT_EXAMPLES["original"][start:end]).strip()
        examples += "\n</example_input>\n"
        
        examples += "<example_output>\n"
        examples += "<S> " + " <S> ".join(PROMPT_SENT_EXAMPLES["normalized"][start:end]).strip()
        examples += "\n</example_output>\n\n"
    
    return prompt + examples


def compile_prompts(
    prompt_name: str,
    system_prompt: str,
    orig_sents: list,
    norm_sents: list,
    batch_size: int
):
    """Compile prompts for model by splitting sentences into batches:
    system prompt + N sentences as user prompt

    Returns:
        list: messages var for OpenRouterAPI request
        list: original sentences (batched)
        list: normalized sentences (batched)
    """
    prompts = []
    orig_sents_batches = []
    norm_sents_batches = []

    assert len(orig_sents) == len(norm_sents)
    num_sentences = len(orig_sents)
    num_batches = ceil(num_sentences / batch_size)

    for i in range(num_batches):
        orig_sents_batch = orig_sents[i*batch_size:(i+1)*batch_size]
        norm_sents_batch = norm_sents[i*batch_size:(i+1)*batch_size]

        orig_sents_batches.append(orig_sents_batch)
        norm_sents_batches.append(norm_sents_batch)

        prompts.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "<S> " + " <S> ".join(orig_sents_batch).strip()}
        ])

    return prompts, orig_sents_batches, norm_sents_batches


def send_request(
    model: str,
    prompt: list, 
    reasoning: None | str, 
    max_retries: int = 5
):
    completion_tokens = 200 # calculated based on average completion tokens
    json_dict = {
        "model": model,
        "messages": prompt,
        "max_tokens": completion_tokens
    }
    if reasoning is not None:
        reasoning_map = {
            "minimal": 4096, "low": 8192, "medium": 16384, "high": 32678
        }
        reasoning_tokens = reasoning_map[reasoning]
        json_dict["max_tokens"] = completion_tokens + reasoning_tokens
        if "anthropic" in model:
            if "4.6" in model:
                # Adaptive reasoning for 4.6+ models
                json_dict["reasoning"] = {
                    "enabled": True
                }
            else:
                json_dict["reasoning"] = {
                    "enabled": True,
                    "max_tokens": reasoning_tokens
                }
        elif "openai" in model.lower() or "google" in model.lower():
            json_dict["reasoning"] = {
                "effort": reasoning
            }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url=f"{settings.OPENROUTER_API_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                    "Content-Type": settings.CONTENT_TYPE,
                },
                json=json_dict,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        
        # Catch network errors (like RemoteDisconnected) or JSON decoding errors from bad gateways
        except (requests.exceptions.RequestException, ValueError) as e:
            if attempt == max_retries - 1:
                print(f"\nRequest failed after {max_retries} attempts.")
                raise  # Re-raise the exception if we've exhausted all retries
            
            sleep_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s...
            print(f"\nNetwork error: {e}. Retrying in {sleep_time} seconds (Attempt {attempt + 1}/{max_retries})...")
            time.sleep(sleep_time)


def separate_reasoning_answer(response_text: str) -> tuple[str, str]:
    """Separate reasoning from answer in response text.

    Looks for <reasoning>...</reasoning> and <answer>...</answer> blocks.
    If not found, returns empty reasoning and full text as answer.

    Args:
        response_text: Full response text from model

    Returns: A tuple(reasoning, answer)
    """
    if not response_text:
        return "", ""
    
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response_text, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    answer = answer_match.group(1).strip() if answer_match else response_text.strip()

    return reasoning, answer


def batch2sents(batch: str) -> list[str]:
    """Process batch response into individual sentences. 
    Splits by <S> separator and strips whitespace.

    Args:
        batch: Text containing sentences separated by <S> (starting from <S>)

    Returns:
        sents_b (list): List of individual sentences from batch
    """
    assert "<S>" in batch
    
    sentences = batch.split("<S>")
    return [sent.strip() for sent in sentences if sent.strip()]


def validate_answer(
    answer: str,
    orig_sents_b: List[str],
    validation_metrics: dict
) -> Tuple[List[str], List[int]]:    
    try:
        pred_sents_b = batch2sents(answer)
    except AssertionError:
        # In case of nonsense answer
        pred_sents_b = [" "] * len(orig_sents_b)
        
    if len(pred_sents_b) < len(orig_sents_b):
        pred_sents_b += [" "] * (len(orig_sents_b) - len(pred_sents_b))
    elif len(pred_sents_b) > len(orig_sents_b):
        pred_sents_b = pred_sents_b[:len(orig_sents_b)]
        
    valid_indices = []
    for i, sents_pair in enumerate(zip(pred_sents_b, orig_sents_b)):
        pred, orig = sents_pair
        if len(pred.split()) == len(orig.split()):
            valid_indices.append(i)
            
    validation_metrics["val_sents"] += len(valid_indices)
    validation_metrics["inval_sents"] += len(pred_sents_b) - len(valid_indices)
    
    return pred_sents_b, valid_indices


def run_evaluation_loop(
    cfg, 
    prompts, 
    orig_sents_bs, 
    norm_sents_bs,
    original, 
    normalized
):
    predicted, valid_predicted, valid_predicted_indices = [], [], []
    total_usage = {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0
    }
    validation_metrics = {
        "total_sents": 0, "val_sents": 0, "inval_sents": 0, "val_rate": 0.0, "error_batches": 0
    }
    batch_offset = 0
    responses_log = []

    run_desc = f"{cfg.model.split('/')[-1]} | bs={cfg.batch_size} | {cfg.prompt}"
    for prompt, orig_b, norm_b in tqdm(
        zip(prompts, orig_sents_bs, norm_sents_bs),
        total=len(prompts), desc=run_desc, file=sys.stdout
    ):
        response = send_request(cfg.model, prompt, cfg.reasoning)
        choices = response.get("choices")
        # In case of model returning error instead of an answer
        if not choices:
            print(f"\n[SKIPPED BATCH] response: {response}")
            validation_metrics["error_batches"] += 1
            validation_metrics["inval_sents"] += len(orig_b)
            predicted.extend([" "] * len(orig_b))
            batch_offset += len(orig_b)
            responses_log.append(("", "", ""))
            log_step_metrics(
                predicted, valid_predicted, valid_predicted_indices,
                normalized, original, total_usage, validation_metrics
            )
            continue
        
        choice = response["choices"][0]
        response_text = choice["message"]["content"] or ""
        reasoning_internal = choice["message"].get("reasoning") or ""  # plain text if available, None if encrypted

        reasoning_from_tags, answer = separate_reasoning_answer(response_text)
        reasoning = reasoning_internal or reasoning_from_tags  # fallback to tag-based
        
        responses_log.append(
            (response_text, reasoning, answer)
        )

        usage = response.get("usage", {})
        for key in total_usage:
            total_usage[key] += usage.get(key, 0)

        pred_sents_b, valid_indices = validate_answer(answer, orig_b, validation_metrics)
        predicted.extend(pred_sents_b)

        for local_idx in valid_indices:
            global_idx = batch_offset + local_idx
            valid_predicted_indices.append(global_idx)
            valid_predicted.append(pred_sents_b[local_idx])

        batch_offset += len(pred_sents_b)

        log_step_metrics(
            predicted, valid_predicted, valid_predicted_indices,
            normalized, original, total_usage, validation_metrics
        )

    return predicted, valid_predicted, valid_predicted_indices, total_usage, validation_metrics, responses_log


def log_step_metrics(
    predicted, 
    valid_predicted, 
    valid_predicted_indices,
    normalized, 
    original, 
    total_usage, 
    validation_metrics
):
    current_total = len(predicted)
    val_rate = validation_metrics["val_sents"] / current_total if current_total > 0 else 0.0

    step_logs = {
        **total_usage,
        "val_sents": validation_metrics["val_sents"],
        "inval_sents": validation_metrics["inval_sents"],
        "val_rate": val_rate,
        "error_batches": validation_metrics["error_batches"],
    }

    current_metrics_all = compute_metrics(
        predicted, normalized[:current_total], original[:current_total]
    )
    for k, v in current_metrics_all.items():
        step_logs[f"all/{k}"] = v

    if valid_predicted:
        current_metrics_valid = compute_metrics(
            valid_predicted,
            [normalized[i] for i in valid_predicted_indices],
            [original[i] for i in valid_predicted_indices],
            valid_alignment=True
        )
        for k, v in current_metrics_valid.items():
            step_logs[f"val/{k}"] = v

    wandb.log(step_logs)
    
    
def log_final_results(cfg, system_prompt, original, normalized,
                      predicted, valid_predicted, valid_predicted_indices,
                      total_usage, validation_metrics, responses_log):
    validation_metrics["total_sents"] = len(original)
    validation_metrics["val_rate"] = validation_metrics["val_sents"] / len(original)

    metrics_all = compute_metrics(predicted, normalized, original)
    metrics_valid = compute_metrics(
        valid_predicted,
        [normalized[i] for i in valid_predicted_indices],
        [original[i] for i in valid_predicted_indices],
        valid_alignment=True
    )

    # Console summary
    print("=="*20)
    print("SUMMARY".center(40))
    print(f"\ntotal tokens: {total_usage['total_tokens']}")
    print(f"val rate: {validation_metrics['val_rate']}\n")
    print("metrics_all:")
    for k, v in metrics_all.items():
        print(f"    {k}: {v}")
    print("\nmetrics_valid:")
    for k, v in metrics_valid.items():
        print(f"    {k}: {v}")
    print("=="*20)

    # W&B summary tables
    results_to_log = {
        "usage_metrics": ("", total_usage),
        "validation_metrics": ("", validation_metrics),
        "prediction_metrics_all": ("all/", metrics_all),
        "prediction_metrics_valid": ("val/", metrics_valid),
    }
    flat_metrics, metric_tables = {}, {}
    for category, (prefix, metrics) in results_to_log.items():
        table = wandb.Table(columns=["Metric", "Value"])
        for k, v in metrics.items():
            flat_metrics[f"{prefix}{k}"] = v
            table.add_data(k, v)
        metric_tables[f"{category}_table"] = table

    wandb.run.summary.update(flat_metrics)

    # Prompt artifact
    run_id = wandb.run.id
    prompt_artifact = wandb.Artifact(name=f"system_prompt_{run_id}", type="prompt")
    with prompt_artifact.new_file(f"system_prompt_{run_id}.txt", mode="w") as f:
        f.write(system_prompt)
    wandb.log_artifact(prompt_artifact)

    # Predictions CSV artifact
    preds_dir = Path("predictions")
    preds_dir.mkdir(parents=True, exist_ok=True)
    
    # Run description for filename
    model = cfg["model"].split("/")[-1]
    parts = [
        model,
        f"bs{cfg['batch_size']}",
        cfg["prompt"].replace("_", "-"),
    ]
    if "reasoning" in cfg:
        reasoning = cfg["reasoning"]
        reasoning = "" if cfg["reasoning"] is None else cfg["reasoning"]
        parts.append(reasoning)
    desc = "_".join(parts) + ".csv"
    
    csv_path = preds_dir / f"{desc}_{run_id}.csv"
    columns = ["Index", "Original", "Normalized", "Predicted", "Is_Valid"]
    valid_set = set(valid_predicted_indices)
    wb_table = wandb.Table(columns=columns)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i in range(len(original)):
            row = [i, original[i], normalized[i], predicted[i], i in valid_set]
            writer.writerow(row)
            wb_table.add_data(*row)

    csv_artifact = wandb.Artifact(name=f"predictions_csv_{run_id}", type="dataset")
    csv_artifact.add_file(str(csv_path))
    wandb.log_artifact(csv_artifact)
    
    responses_table = wandb.Table(columns=["raw_response", "reasoning", "answer"])
    for entry in responses_log:
        responses_table.add_data(*entry)
    
    wandb.log({
        **metric_tables,
        "predictions_table": wb_table,
        "responses": responses_table
    })


@hydra_main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):    
    wandb.init(
        project="normalization", 
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit="finish_previous",
        name=f"{cfg.model.split('/')[-1]}_bs{cfg.batch_size}_{cfg.prompt}"
    )

    try:
        # Setup
        system_prompt = load_prompt(cfg.prompt, cfg.batch_size)
        _, original, _, normalized = get_train_test(
            n=cfg.num_sentences,
            test_size=cfg.test_size,
            shuffle=True
        )
        prompts, orig_sents_bs, norm_sents_bs = compile_prompts(
            cfg.prompt, system_prompt, original, normalized, cfg.batch_size
        )
        
        # Evaluation loop
        predicted, valid_predicted, valid_predicted_indices, \
            total_usage, validation_metrics, responses_log = run_evaluation_loop(
                cfg, prompts, orig_sents_bs, 
                norm_sents_bs, original, normalized
            )
            
        log_final_results(
            cfg, system_prompt, original, normalized,
            predicted, valid_predicted, valid_predicted_indices,
            total_usage, validation_metrics, responses_log
        )
    except Exception as e:
        import traceback
        print(f"\n[FAILED] {cfg.model} / {cfg.prompt} / bs={cfg.batch_size}")
        traceback.print_exc()
        wandb.log({"failed": True, "error": str(e)})
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()