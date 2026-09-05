from ..agent.registry import register_tool
from ..config.settings import settings
from ..evidence.confidence import compute_confidence
from ..models.geochat import GeoChatModel
from ..preprocessing.raster import raster_to_rgb_image
@register_tool('caption')
def caption_tool(state):
    image=raster_to_rgb_image(state['image_paths'][0],settings.max_image_side)
    model=GeoChatModel.singleton(settings.model_id,settings.base_model_id,settings.model_max_pixels)
    answer,certainty=model.ask_with_certainty(image,'Describe this remote-sensing image. Summarize dominant land cover, vegetation, water, roads, built-up areas and notable spatial patterns. Use only visible evidence.',max_new_tokens=settings.max_new_tokens,temperature=settings.temperature)
    confidence=compute_confidence(base=0.6,model_certainty=certainty,input_quality=state.get('input_quality',1.0))
    return {'answer':answer,'confidence':confidence,'tool':'caption','model':settings.model_id,'evidence':[{'type':'image','path':state['image_paths'][0]}]}
