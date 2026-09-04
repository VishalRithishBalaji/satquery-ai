from pathlib import Path
import rasterio

def validate_image(path: str):
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(path)
    ext=p.suffix.lower()
    if ext not in {'.tif','.tiff','.png','.jpg','.jpeg'}: raise ValueError(f'Unsupported format: {ext}')
    info={'path':str(p.resolve()),'filename':p.name,'extension':ext,'format':'raster'}
    try:
        with rasterio.open(p) as src:
            info.update({'width':src.width,'height':src.height,'count':src.count,'crs':str(src.crs) if src.crs else None,'transform':str(src.transform),'dtype':str(src.dtypes[0]) if src.dtypes else None,'bounds':[src.bounds.left,src.bounds.bottom,src.bounds.right,src.bounds.top]})
    except Exception:
        from PIL import Image
        with Image.open(p) as im: info.update({'width':im.width,'height':im.height,'count':len(im.getbands()),'crs':None,'transform':None,'dtype':str(im.getbands())})
    return info
