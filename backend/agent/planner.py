import re

def infer_task(query: str, image_count: int) -> str:
    q=query.lower().strip()
    if image_count >= 2 and any(x in q for x in ['optical','sar','radar','both sensors','together']): return 'optical_sar'
    if image_count >= 2 and any(x in q for x in ['change','changed','before','after','between','increased','decreased','difference']): return 'change_vqa'
    if any(x in q for x in ['highlight','locate','draw a box','bounding box','where is']): return 'grounding'
    if any(x in q for x in ['describe','scene','land cover','summarize','what is visible']): return 'caption'
    return 'vqa'
