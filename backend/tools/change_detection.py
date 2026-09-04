from ..agent.registry import register_tool
from ..config.settings import settings
from ..preprocessing.alignment import align_pair_to_common_rgb, validate_alignment
from ..evidence.change_map import make_change_map
@register_tool('change_detection')
def change_detection_tool(state):
    if len(state.get('image_paths',[]))!=2: raise ValueError('Change detection requires exactly two images')
    imgs=align_pair_to_common_rgb(state['image_paths']); out=settings.output_path/'change_maps'/f"change_{state['run_id']}.png"
    r=make_change_map(imgs[0],imgs[1],str(out)); frac=r['changed_fraction']
    summary='Localized changes were detected.' if frac<0.45 else 'Large image differences were detected; seasonal, illumination, or alignment effects may contribute.'
    if frac<0.01: summary='Very limited pixel-level change was detected.'
    return {'answer':summary,'confidence':0.58,'tool':'change_detection','model':'OpenCV change-map prototype','evidence':[{'type':'change_map','path':r['path'],'changed_fraction':frac,'regions':r['changed_regions']}]}
