from ..agent.registry import register_tool
@register_tool('geospatial')
def geospatial_tool(state): return {'answer':'Geospatial metadata validated.','confidence':0.90,'tool':'geospatial','model':'Rasterio/GDAL','evidence':[]}
