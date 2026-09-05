from ..agent.query_analysis import QueryIntent, analyze_query
from ..agent.query_aware import run_query_aware_vlm
from ..agent.registry import register_tool
from ..agent.sensitivity import image_set_key
from ..config.settings import settings
from ..models.geochat import GeoChatModel
from ..preprocessing.raster import raster_to_rgb_image


def build_prompt(intent: QueryIntent, query: str, previous_answer: str | None = None, force: bool = False) -> str:
    focus = (
        "Focus specifically on: " + ", ".join(a.replace("_", " ") for a in intent.attributes) + "."
        if intent.attributes else ""
    )
    if force and previous_answer:
        correction = f'\nIMPORTANT: Do not repeat this earlier answer, which was for a DIFFERENT question: "{previous_answer}"\n'
    elif force:
        correction = "\nIMPORTANT: Your previous attempt did not specifically address this question. Do not give a generic description - directly answer what is asked below.\n"
    else:
        correction = ""
    return (
        "You are a remote-sensing VQA assistant. Answer only from visible evidence. "
        "Do not invent objects. Mention uncertainty when the image is insufficient. "
        f"{focus}{correction}\nQuestion: {query}"
    )


@register_tool('vqa')
def vqa_tool(state):
    image_path = state['image_paths'][0]
    image = raster_to_rgb_image(image_path, settings.max_image_side)
    model = GeoChatModel.singleton(settings.model_id, settings.base_model_id, settings.model_max_pixels)
    intent = analyze_query(state['query'], 1)

    generation = run_query_aware_vlm(
        model=model,
        images=[image],
        query=state['query'],
        intent=intent,
        image_key=image_set_key([image_path]),
        prompt_builder=build_prompt,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        base_confidence=0.5,
        input_quality=state.get('input_quality', 1.0),
    )

    return {
        'answer': generation['answer'],
        'confidence': generation['confidence'],
        'tool': 'vqa',
        'model': settings.model_id,
        'task_intent': {'operation': intent.operation, 'attributes': intent.attributes},
        'query_grounding_score': generation['query_grounding_score'],
        'model_certainty': generation['model_certainty'],
        'low_query_sensitivity': generation['low_query_sensitivity'],
        'retried_for_query_sensitivity': generation['retried_for_query_sensitivity'],
        'evidence': [{'type': 'image', 'path': image_path}],
    }
