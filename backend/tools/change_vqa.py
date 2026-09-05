from ..agent.query_analysis import QueryIntent, analyze_query
from ..agent.query_aware import run_query_aware_vlm
from ..agent.registry import register_tool
from ..agent.sensitivity import image_set_key
from ..config.settings import settings
from ..evidence.confidence import change_evidence_strength
from ..models.geochat import GeoChatModel
from ..preprocessing.raster import raster_to_rgb_image


def build_prompt(intent: QueryIntent, query: str, previous_answer: str | None = None, force: bool = False) -> str:
    focus = (
        "Pay particular attention to: " + ", ".join(a.replace("_", " ") for a in intent.attributes) + "."
        if intent.attributes else ""
    )
    if force and previous_answer:
        correction = f'\nIMPORTANT: Do not repeat this earlier answer, which was for a DIFFERENT question: "{previous_answer}"\n'
    elif force:
        correction = "\nIMPORTANT: Your previous attempt did not specifically address this question. Do not give a generic description - directly answer what is asked below.\n"
    else:
        correction = ""
    return (
        "The first image is BEFORE and the second is AFTER. Compare them carefully. "
        "Answer the question using visible evidence and distinguish likely real changes from "
        f"seasonal, illumination or alignment differences. {focus}{correction}\nQuestion: {query}"
    )


def _prior_change_fraction(state) -> float | None:
    for result in state.get('tool_results', []) or []:
        if result.get('tool') != 'change_detection':
            continue
        for item in result.get('evidence', []):
            if 'changed_fraction' in item:
                return float(item['changed_fraction'])
    return None


@register_tool('change_vqa')
def change_vqa_tool(state):
    if len(state.get('image_paths', [])) != 2:
        raise ValueError('Change-VQA requires exactly two images')

    images = [raster_to_rgb_image(p, settings.max_image_side) for p in state['image_paths']]
    model = GeoChatModel.singleton(settings.model_id, settings.base_model_id, settings.model_max_pixels)
    intent = analyze_query(state['query'], 2)

    generation = run_query_aware_vlm(
        model=model,
        images=images,
        query=state['query'],
        intent=intent,
        image_key=image_set_key(state['image_paths']),
        prompt_builder=build_prompt,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        base_confidence=0.5,
        evidence_strength=change_evidence_strength(_prior_change_fraction(state)),
        input_quality=state.get('input_quality', 1.0),
    )

    return {
        'answer': generation['answer'],
        'confidence': generation['confidence'],
        'tool': 'change_vqa',
        'model': settings.model_id,
        'task_intent': {'operation': intent.operation, 'attributes': intent.attributes},
        'query_grounding_score': generation['query_grounding_score'],
        'model_certainty': generation['model_certainty'],
        'low_query_sensitivity': generation['low_query_sensitivity'],
        'retried_for_query_sensitivity': generation['retried_for_query_sensitivity'],
        'evidence': [{'type': 'temporal_pair', 'before': state['image_paths'][0], 'after': state['image_paths'][1]}],
    }
