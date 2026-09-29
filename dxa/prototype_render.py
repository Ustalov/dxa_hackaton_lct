"""Only user-authorized overlays; no general bone segmentation or object heatmaps."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render(raw, region, data, masks, output, spacing_yx=(1.05,.6)):
    scale = 3; h,w = raw.shape
    rgb = np.repeat(raw[:,:,None], 3, axis=2).astype(float)
    if region == 'hip':
        if 'margins' in data:
            sy,sx=spacing_yx
            # Fractional edge pixels preserve the requested physical band thickness.
            top=np.clip(30/sy-np.arange(h),0,1)[:,None]
            bottom=np.clip(30/sy-np.arange(h)[::-1],0,1)[:,None]
            bands=[(top,(0,164,255)),(bottom,(164,100,255))]
            lateral=data.get('margins',{}).get('femur_lateral_margin',{}).get('target_image_edge') or data.get('lateral_image_side')
            if lateral in ('left','right'):
                distance=np.arange(w) if lateral=='left' else np.arange(w)[::-1]
                bands.append((np.clip(20/sx-distance,0,1)[None,:],(0,210,175)))
            for coverage,color in bands:
                alpha=.28*coverage[:,:,None]
                rgb=rgb*(1-alpha)+np.array(color)*alpha
        for key, color, alpha in [('lt',(255,90,100),.4), ('protrusion',(255,215,65),.65)]:
            mask = masks.get(key)
            if mask is not None:
                rgb[mask] = rgb[mask]*(1-alpha) + np.array(color)*alpha
    image = Image.fromarray(np.uint8(rgb)).resize((w*scale,h*scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11*scale)
    def point(p): return tuple(float(v)*scale for v in p)
    def text(p, s, color):
        box=draw.textbbox((0,0),s,font=font)
        x=min(max(3,p[0]*scale),max(3,w*scale-(box[2]-box[0])-4))
        y=min(max(3,p[1]*scale),max(3,h*scale-(box[3]-box[1])-10))
        draw.text((x,y),s,font=font,fill=color,stroke_width=scale,stroke_fill=(15,20,25))
    def double_arrow(a,b):
        a=np.array(point(a));b=np.array(point(b));length=float(np.linalg.norm(b-a))
        if length<1:return
        direction=(b-a)/length;normal=np.array([-direction[1],direction[0]])
        head=min(4*scale,length/3)
        paths=[[tuple(a),tuple(b)]]
        for tip,inward in ((a,direction),(b,-direction)):
            paths.append([tuple(tip+inward*head+normal*head*.65),tuple(tip),
                          tuple(tip+inward*head-normal*head*.65)])
        for width,color in ((3*scale,(15,20,25)),(scale,(245,250,255))):
            for path in paths:draw.line(path,fill=color,width=width)
    if region == 'hip' and 'margins' in data:
        top_end=min(30/spacing_yx[0],h-1)
        bottom_start=max(0,h-30/spacing_yx[0])
        double_arrow((w/2,0),(w/2,top_end))
        text((w/2+6,top_end/2-7),'30 мм',(245,250,255))
        double_arrow((w/2,bottom_start),(w/2,h-1))
        text((w/2+6,(bottom_start+h-1)/2-7),'30 мм',(245,250,255))
        if lateral in ('left','right'):
            x1,x2=(0,min(20/spacing_yx[1],w-1)) if lateral=='left' else (max(0,w-20/spacing_yx[1]),w-1)
            double_arrow((x1,h/2),(x2,h/2))
            label_width=draw.textlength('20 мм',font=font)/scale
            text(((x1+x2-label_width)/2,h/2+5),'20 мм',(245,250,255))
    if region != 'hip':
        axis=data.get('axis',{});pts=axis.get('endpoints_px')
        if pts:
            bottom,top=pts;color=(255,200,70)
            draw.line([point(bottom),point(top)],fill=color,width=2*scale)
            for y in np.arange(max(0,top[1]),min(h-1,bottom[1]),7):
                draw.line([point([bottom[0],y]),point([bottom[0],min(y+3,bottom[1])])],fill=(235,235,235),width=scale)
            text((top[0]+5,top[1]+6),f"{axis['value']:.1f}°",color)
        for value in data.get('iliac',{}).values():
            curve=value.get('upper_contour',[])
            for a,b in zip(curve,curve[1:]):
                if b[0]-a[0] <= 1 and abs(b[1]-a[1])*spacing_yx[0] <= 5:
                    draw.line([point(a),point(b)],fill=(75,245,155),width=2*scale)
    image.save(output)
