from ..agent.registry import register_tool
from ..config.settings import settings
from ..models.geochat import GeoChatModel
from ..preprocessing.raster import raster_to_rgb_image
@register_tool('change_vqa')
def change_vqa_tool(state):
    if len(state.get('image_paths',[]))!=2: raise ValueError('Change-VQA requires exactly two images')
    images=[raster_to_rgb_image(p,settings.max_image_side) for p in state['image_paths']]
    model=GeoChatModel.singleton(settings.model_id,settings.base_model_id,settings.model_max_pixels)
    answer=model.ask_multiple(images,'The first image is BEFORE and the second is AFTER. Compare them carefully. Answer the question using visible evidence and distinguish likely real changes from seasonal, illumination or alignment differences. Question: '+state['query'],max_new_tokens=settings.max_new_tokens,temperature=settings.temperature)
    return {'answer':answer,'confidence':0.64,'tool':'change_vqa','model':settings.model_id,'evidence':[{'type':'temporal_pair','before':state['image_paths'][0],'after':state['image_paths'][1]}]}
