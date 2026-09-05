"""Shared query-aware VLM orchestration used by every tool that calls
GeoChatModel (VQA, change-VQA, Optical-SAR fusion).

This is the concrete fix for "same image, different queries, same answer":
1. build a task-specific prompt (per-tool `prompt_builder`, not one generic
   paragraph for every query),
2. generate an answer and score how well it engages the specific query,
3. check it against recent answers for this same imagery - if a materially
   different query produced a near-identical answer, that is flagged as
   low query sensitivity,
4. when the answer is poorly grounded or flagged, retry once with a
   corrective prompt that explicitly names the earlier answer and forbids
   repeating it,
5. fold every signal gathered along the way into one evidence-based
   confidence score instead of a fixed constant.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..evidence.confidence import compute_confidence
from .query_analysis import QueryIntent
from .response_validator import grounding_score as compute_grounding_score
from .sensitivity import check_sensitivity, remember

PromptBuilder = Callable[..., str]

_RETRY_GROUNDING_FLOOR = 0.25
_SENSITIVITY_PENALTY = 0.2


def run_query_aware_vlm(
    *,
    model: Any,
    images: list,
    query: str,
    intent: QueryIntent,
    image_key: str,
    prompt_builder: PromptBuilder,
    max_new_tokens: int,
    temperature: float,
    base_confidence: float = 0.5,
    evidence_strength: float | None = None,
    cross_modal_agreement: float | None = None,
    input_quality: float = 1.0,
) -> dict:
    def generate(prompt: str) -> tuple[str, float | None]:
        if len(images) > 1:
            return model.ask_multiple_with_certainty(images, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        return model.ask_with_certainty(images[0], prompt, max_new_tokens=max_new_tokens, temperature=temperature)

    answer, certainty = generate(prompt_builder(intent, query))
    grounding = compute_grounding_score(answer, intent)
    sensitivity = check_sensitivity(image_key, intent, answer)

    retried = False
    if sensitivity["low_sensitivity"] or grounding < _RETRY_GROUNDING_FLOOR:
        corrective_prompt = prompt_builder(
            intent,
            query,
            previous_answer=sensitivity.get("most_similar_answer"),
            force=True,
        )
        retry_answer, retry_certainty = generate(corrective_prompt)
        retry_grounding = compute_grounding_score(retry_answer, intent)
        if retry_grounding > grounding:
            answer, certainty, grounding = retry_answer, retry_certainty, retry_grounding
            retried = True

    remember(image_key, intent, query, answer)

    low_sensitivity_final = bool(sensitivity["low_sensitivity"] and not retried)

    confidence = compute_confidence(
        base=base_confidence,
        grounding_score=grounding,
        model_certainty=certainty,
        evidence_strength=evidence_strength,
        cross_modal_agreement=cross_modal_agreement,
        sensitivity_penalty=_SENSITIVITY_PENALTY if low_sensitivity_final else 0.0,
        input_quality=input_quality,
    )

    return {
        "answer": answer,
        "confidence": confidence,
        "query_grounding_score": round(grounding, 3),
        "model_certainty": round(certainty, 3) if certainty is not None else None,
        "low_query_sensitivity": low_sensitivity_final,
        "retried_for_query_sensitivity": retried,
    }
