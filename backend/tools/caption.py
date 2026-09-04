from ..agent.registry import register_tool
from ..config.settings import settings
from ..models.geochat import GeoChatModel
from ..preprocessing.raster import raster_to_rgb_image
@register_tool('caption')
def caption_tool(state):
    image=raster_to_rgb_image(state['image_paths'][0],settings.max_image_side)
    model=GeoChatModel.singleton(settings.model_id,settings.base_model_id,settings.model_max_pixels)
    answer=model.ask(image,'Describe this remote-sensing image. Summarize dominant land cover, vegetation, water, roads, built-up areas and notable spatial patterns. Use only visible evidence.',max_new_tokens=settings.max_new_tokens,temperature=settings.temperature)
    return {'answer':answer,'confidence':0.68,'tool':'caption','model':settings.model_id,'evidence':[{'type':'image','path':state['image_paths'][0]}]}
