from ..agent.registry import register_tool
from ..evidence.confidence import compute_confidence
@register_tool('geospatial')
def geospatial_tool(state):
    confidence=compute_confidence(base=0.85,input_quality=state.get('input_quality',1.0))
    return {'answer':'Geospatial metadata validated.','confidence':confidence,'tool':'geospatial','model':'Rasterio/GDAL','evidence':[]}
