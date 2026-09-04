from ..agent.registry import register_tool
from ..config.settings import settings
from ..models.geochat import GeoChatModel
from ..preprocessing.raster import raster_to_rgb_image
@register_tool('vqa')
def vqa_tool(state):
    image=raster_to_rgb_image(state['image_paths'][0],settings.max_image_side)
    model=GeoChatModel.singleton(settings.model_id,settings.base_model_id,settings.model_max_pixels)
    answer=model.ask(image,'You are a remote-sensing VQA assistant. Answer only from visible evidence. Do not invent objects. Mention uncertainty when the image is insufficient. Question: '+state['query'],max_new_tokens=settings.max_new_tokens,temperature=settings.temperature)
    return {'answer':answer,'confidence':0.70,'tool':'vqa','model':settings.model_id,'evidence':[{'type':'image','path':state['image_paths'][0]}]}
