from .planner import infer_task
TOOL_MAP={'vqa':['vqa'],'caption':['caption'],'grounding':['grounding'],'change_vqa':['change_detection','change_vqa'],'optical_sar':['optical_sar']}
def route(query,image_count):
    task=infer_task(query,image_count); return task, TOOL_MAP.get(task,['vqa'])
