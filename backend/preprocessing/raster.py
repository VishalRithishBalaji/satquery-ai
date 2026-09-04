from PIL import Image
import numpy as np

def raster_to_rgb_image(path: str, max_side: int=1536) -> Image.Image:
    try:
        import rasterio
        with rasterio.open(path) as src:
            arr=src.read(out_dtype='float32')
        if arr.shape[0]>=3: arr=np.moveaxis(arr[:3],0,-1)
        else:
            band=arr[0]; arr=np.stack([band,band,band],axis=-1)
        lo=np.nanpercentile(arr,2,axis=(0,1),keepdims=True); hi=np.nanpercentile(arr,98,axis=(0,1),keepdims=True)
        arr=np.clip((arr-lo)/(hi-lo+1e-6),0,1); out=(arr*255).astype('uint8'); im=Image.fromarray(out)
    except Exception:
        im=Image.open(path).convert('RGB')
    if max(im.size)>max_side:
        scale=max_side/max(im.size); im=im.resize((max(1,int(im.width*scale)),max(1,int(im.height*scale))),Image.Resampling.LANCZOS)
    return im
