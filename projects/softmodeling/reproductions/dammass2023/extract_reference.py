"""Digitize published MODEL curves, not experimental raw data.

Requires optional PyMuPDF. Fig.5 uses PDF Bezier paths; Fig.7 uses specified
plot axes and the saturated blue/orange model strokes in the rendered PDF.
No reproduction result is read here. Source PDF coordinates are retained.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module-dir',type=Path)
    args=parser.parse_args()
    if args.module_dir:sys.path.insert(0,str(args.module_dir.resolve()))
    import pymupdf
    source=Path('references/updated phase.pdf');doc=pymupdf.open(source)
    out=Path(__file__).parent/'reference';out.mkdir(exist_ok=True)
    metadata=dict(source_pdf=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_type='published model curves; not experimental measurements',
        fig5=dict(page=12,path_indices=[33,108,185],axis_pdf=dict(x1=104.357,x2_5=267.446,y0=179.100,y20=67.867)),
        fig7=dict(page=14,panel='A',axis_pdf=dict(x0=123.771,x20=248.037,y0=175.895,y50=64.001),
            method='saturated curve-color pixels; upper/lower stroke ordinates retained per column; prepeak upper envelope sampled at 1 mm',
            estimated_reading_uncertainty_force_n=1.,estimated_reading_uncertainty_opening_mm=.4,
            uncertainty_note='Approximate visual reading uncertainty including black marker occlusion; not a statistical confidence interval'))
    rows=[];raw=[]
    for index,rate in [(33,.3),(108,.044),(185,.0056)]:
        path=doc[11].get_drawings()[index]
        raw.append(dict(rate=rate,items=[[item[0],*[list(p) for p in item[1:]]] for item in path['items']]))
        points=[]
        for item in path['items']:
            if item[0]=='l':
                a,b=map(np.array,item[1:]);t=np.linspace(0,1,20)[:,None];points.extend((1-t)*a+t*b)
            elif item[0]=='c':
                a,b,c,d=map(np.array,item[1:]);t=np.linspace(0,1,100)[:,None]
                points.extend((1-t)**3*a+3*(1-t)**2*t*b+3*(1-t)*t*t*c+t**3*d)
            else:raise ValueError('Unexpected path primitive')
        points=np.array(points)
        for x,y in points:
            rows.append(dict(stretch_rate_s=rate,stretch=1+(x-104.357)*1.5/(267.446-104.357),
                stress_mpa=(179.100-y)*20/(179.100-67.867),pdf_x=x,pdf_y=y))
    with (out/'fig5.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'fig5_vector_paths.json').write_text(json.dumps(raw,indent=2)+'\n')
    scale=6
    pix=doc[13].get_pixmap(matrix=pymupdf.Matrix(scale,scale))
    rgb=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,pix.n)[:,:,:3]/255
    rows=[];landmarks=[]
    for rate,color in [(25,[.1215686,.4666667,.7058671]),(200,[1.,.4980392,.054902])]:
        mask=np.max(np.abs(rgb-np.array(color)),axis=2)<.012
        mask[:int(60*scale)]=False;mask[int(177*scale):]=False
        mask[:,:int(123.5*scale)]=False;mask[:,int(249*scale):]=False
        columns=[]
        for column in np.flatnonzero(mask.any(axis=0)):
            locations=np.flatnonzero(mask[:,column]);x=(column+.5)/scale
            u=(x-123.771)*20/(248.037-123.771)
            upper=(175.895-(locations.min()+.5)/scale)*50/(175.895-64.001)
            lower=(175.895-(locations.max()+.5)/scale)*50/(175.895-64.001)
            if u>=0:columns.append((u,upper,lower,x))
        data=np.array(columns);peak=data[np.argmax(data[:,1])]
        last=data[-1]
        landmarks.append(dict(speed_mm_min=rate,peak_opening_mm=peak[0],peak_force_n=peak[1],
            last_opening_mm=last[0],last_lower_force_n=last[2]))
        for u,upper,lower,x in data:
            rows.append(dict(speed_mm_min=rate,opening_mm=u,upper_force_n=upper,lower_force_n=lower,pdf_x=x))
    with (out/'fig7_stroke_envelopes.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'fig7_landmarks.json').write_text(json.dumps(landmarks,indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(landmarks,indent=2))


if __name__=='__main__':main()
