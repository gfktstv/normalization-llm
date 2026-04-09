"""Metrics computation for sentence normalization evaluation"""
import re
from typing import List, Dict, Tuple
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from nltk.metrics.distance import edit_distance


# ============================================================================
# SENTENCE-WISE METRICS
# ============================================================================

def rouge1_score(predictions: List[str], targets: List[str]) -> float:
    """Calculate ROUGE-1 F1 score

    Args:
        predictions: List of predicted sentences
        targets: List of target sentences

    Returns:
        Average ROUGE-1 F1 score (0-100)
    """
    if not predictions:
        return 0.0
    
    class CyrillicTokenizer:
        def tokenize(self, text):
            return re.findall(r'\w+', text, re.UNICODE)

    scorer = rouge_scorer.RougeScorer(
        ['rouge1'],
        use_stemmer=False,
        tokenizer=CyrillicTokenizer()
    )
    scores = [scorer.score(targ, pred)['rouge1'].fmeasure * 100
              for pred, targ in zip(predictions, targets)]
    return sum(scores) / len(scores) if scores else 0.0


def bleu_score(predictions: List[str], targets: List[str]) -> float:
    """Calculate BLEU score

    Args:
        predictions: List of predicted sentences
        targets: List of target sentences

    Returns:
        Average BLEU score (0-100)
    """
    if not predictions:
        return 0.0

    bleu = BLEU()
    score = bleu.corpus_score(predictions, [targets])
    return score.score


def sentence_levenshtein(
    predictions: List[str], 
    targets: List[str],
    normalization: None | str = None
) -> float:
    """Mean value across all levenstein distances between 
    each pair of reference and generated sentences. 
    Optionally normalized by maximum word count (normalization="max")
    or target word count (normalization="target") (WER)

    Args:
        predictions: List of predicted normalized sentences
        targets: List of target normalized sentences
        normalization (None | str): normalization factor, "max" or "target"

    Returns:
        float: mean distance
    """
    if normalization is not None:
        assert (normalization == "max" or normalization == "target")
    if not predictions:
        return 0.0
    
    distances = []
    for pred, targ in zip(predictions, targets):
        distance = edit_distance(pred.split(), targ.split())
        if normalization is None:
            distances.append(distance)
        else:
            if normalization == "max":
                norm_factor = max(len(pred.split()), len(targ.split()))
            elif normalization == "target":
                norm_factor = len(targ.split())
            
            normalized = distance / norm_factor if norm_factor > 0 else 0.0
            distances.append(normalized)

    return sum(distances) / len(distances) if distances else 0.0


def exact_match(predictions: List[str], targets: List[str]) -> float:
    """Percentage of predictions that exactly match the target

    Args:
        predictions: List of predicted sentences
        targets: List of target sentences

    Returns:
        float: Exact match score (0-100)
    """
    if not predictions:
        return 0.0
    
    matches = sum(pred == targ for pred, targ in zip(predictions, targets))
    return matches / len(predictions) * 100


# ============================================================================
# TOKEN-WISE METRICS
# ============================================================================

def classify_tokens_definition1(predictions: List[str], targets: List[str], originals: List[str]) -> Tuple[int, int, int, int]:
    """Classify tokens using Definition 1 (based on whether token was changed)

    Definition 1 - simple change detection (without checking correctness):
    - TP: orig != targ (change needed) AND orig != pred (model applied change)
    - TN: orig == targ (no change needed) AND orig == pred (model didn't change)
    - FP: orig == targ (no change needed) AND orig != pred (model incorrectly changed)
    - FN: orig != targ (change needed) AND orig == pred (model didn't apply change)

    Args:
        predictions: List of predicted normalized sentences
        targets: List of target normalized sentences
        originals: List of original sentences

    Returns:
        Tuple of (TP, TN, FP, FN) counts
    """
    tp, tn, fp, fn = 0, 0, 0, 0

    for pred, targ, orig in zip(predictions, targets, originals):
        pred_tokens = pred.split()
        targ_tokens = targ.split()
        orig_tokens = orig.split()

        # Position-based alignment
        for i in range(max(len(pred_tokens), len(targ_tokens), len(orig_tokens))):
            pred_token = pred_tokens[i] if i < len(pred_tokens) else ''
            targ_token = targ_tokens[i] if i < len(targ_tokens) else ''
            orig_token = orig_tokens[i] if i < len(orig_tokens) else ''

            needs_change = orig_token != targ_token
            was_changed = orig_token != pred_token

            if needs_change and was_changed:
                tp += 1
            elif not needs_change and not was_changed:
                tn += 1
            elif not needs_change and was_changed:
                fp += 1
            else:  # needs_change and not was_changed
                fn += 1

    return tp, tn, fp, fn


def classify_tokens_definition2(predictions: List[str], targets: List[str], originals: List[str]) -> Tuple[int, int, int, int]:
    """Classify tokens using Definition 2 (based on correctness)

    Definition 2 - change correctness checking:
    - TP: orig != targ (change needed) AND pred == targ (model normalized correctly)
    - TN: orig == targ (no change needed) AND pred == orig (model didn't change)
    - FP: orig == targ (no change needed) AND pred != orig (model incorrectly changed)
    - FN: orig != targ (change needed) AND pred != targ (model changed incorrectly)

    Args:
        predictions: List of predicted normalized sentences
        targets: List of target normalized sentences
        originals: List of original sentences

    Returns:
        Tuple of (TP, TN, FP, FN) counts
    """
    tp, tn, fp, fn = 0, 0, 0, 0

    for pred, targ, orig in zip(predictions, targets, originals):
        pred_tokens = pred.split()
        targ_tokens = targ.split()
        orig_tokens = orig.split()

        # Position-based alignment
        for i in range(max(len(pred_tokens), len(targ_tokens), len(orig_tokens))):
            pred_token = pred_tokens[i] if i < len(pred_tokens) else ''
            targ_token = targ_tokens[i] if i < len(targ_tokens) else ''
            orig_token = orig_tokens[i] if i < len(orig_tokens) else ''

            needs_change = orig_token != targ_token

            if needs_change:
                # Token needs normalization
                if pred_token == targ_token:
                    tp += 1  # Normalized correctly
                else:
                    fn += 1  # Normalized incorrectly
            else:
                # Token doesn't need normalization
                if pred_token == orig_token:
                    tn += 1  # Didn't change (correct)
                else:
                    fp += 1  # Changed when it shouldn't (incorrect)

    return tp, tn, fp, fn


def token_metrics(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    """Calculate token-wise metrics from confusion matrix

    Args:
        tp, tn, fp, fn: Confusion matrix counts

    Returns:
        Dictionary with Accuracy, Precision, Recall, F1-score
    """
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy * 100,
        "specificity": specificity * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
    }
    
def token_levenstein(
    predictions: List[str], 
    targets: List[str],
    normalization: None | str = None
) -> float:
    """Mean value across all levenstein distances between 
    each pair of reference and generated tokens. 
    Optionally normalized by maximum character count (normalization="max")
    or target character count (normalization="target") (WER)
    
    Calculated only for aligned by token sentences

    Args:
        predictions: List of predicted normalized sentences
        targets: List of target normalized sentences
        normalization (None | str): normalization factor, "max" or "target"

    Returns:
        float: mean distance
    """
    if normalization is not None:
        assert (normalization == "max" or normalization == "target")
    if not predictions:
        return 0.0
    
    predicted_tokens, target_tokens = [], []
    distances = []
    for pred, targ in zip(predictions, targets):
        preds, targs = pred.split(), targ.split()
        if len(preds) == len(targs):
            predicted_tokens.extend(preds)
            target_tokens.extend(targs)
            
    if not predicted_tokens:
        return 0.0
    
    for pred, targ in zip(predicted_tokens, target_tokens):
        distance = edit_distance(pred, targ)
        if normalization is None:
            distances.append(distance)
        else:
            if normalization == "max":
                norm_factor = max(len(pred), len(targ))
            elif normalization == "target":
                norm_factor = len(targ)
                
            normalized = distance / norm_factor if norm_factor > 0 else 0.0
            distances.append(normalized)
            
    return sum(distances) / len(distances) if distances else 0.0


def compute_metrics(
    predictions: List[str],
    targets: List[str],
    originals: List[str],
    valid_alignment: bool = False,
    lowercase: bool = True
) -> Dict[str, float]:
    """Compute all evaluation metrics

    Args:
        predictions: List of predicted sentences
        targets: List of target sentences
        originals: List of original sentences (required for token-wise metrics)

    Returns:
        Dictionary with all computed metrics
    """
    if lowercase:
        predictions = [s.lower() for s in predictions]
        targets = [s.lower() for s in targets]
        originals = [s.lower() for s in originals]
    
    metrics = {
        # Sentence-wise metrics
        "rouge1": rouge1_score(predictions, targets),
        "bleu": bleu_score(predictions, targets),
        "mld-s": sentence_levenshtein(predictions, targets),
        "nmld-s": sentence_levenshtein(predictions, targets, normalization="max"),
        "wer": sentence_levenshtein(predictions, targets, normalization="target"),
        "em": exact_match(predictions, targets)
    }

    # Token-wise metrics - Definition 1 (change detection)
    tp1, tn1, fp1, fn1 = classify_tokens_definition1(predictions, targets, originals)
    metrics.update({f"1/{k}": v for k, v in token_metrics(tp1, tn1, fp1, fn1).items()})

    # Token-wise metrics - Definition 2 (correctness checking)
    tp2, tn2, fp2, fn2 = classify_tokens_definition2(predictions, targets, originals)
    metrics.update({f"2/{k}": v for k, v in token_metrics(tp2, tn2, fp2, fn2).items()})
    
    if valid_alignment:
        # Token-wise metrics - Levenstein
        metrics.update({
            "mld-t": token_levenstein(predictions, targets),
            "nmld-t": token_levenstein(predictions, targets, normalization="max"),
            "cer": token_levenstein(predictions, targets, normalization="target")
        })

    return metrics