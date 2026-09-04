from pathlib import Path
import cv2, numpy as np

def make_change_map(before,after,out_path):
    a=np.array(before.convert('RGB')); b=np.array(after.convert('RGB')); b=cv2.resize(b,(a.shape[1],a.shape[0]))
    d=cv2.absdiff(a,b); gray=cv2.cvtColor(d,cv2.COLOR_RGB2GRAY); blur=cv2.GaussianBlur(gray,(5,5),0); _,mask=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); mask=cv2.dilate(mask,np.ones((3,3),np.uint8),iterations=1)
    overlay=a.copy(); overlay[mask>0]=[255,64,64]; blend=cv2.addWeighted(a,0.65,overlay,0.35,0); Path(out_path).parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(out_path,cv2.cvtColor(blend,cv2.COLOR_RGB2BGR))
    changed=float((mask>0).mean()); n,labels,stats,cent=cv2.connectedComponentsWithStats(mask); regions=[]
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if area>=max(25,0.001*mask.size): regions.append({'bbox':[int(x),int(y),int(x+w),int(y+h)],'area_pixels':int(area)})
    return {'path':str(out_path),'changed_fraction':changed,'changed_regions':regions[:50]}
