"""Plot every available AT1 probe on common axes/scales; no displacement scaling."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_support import write_json


def render_continuous(source, output, result, *, movie=True):
    """Saved damaged stresses only: never recompute an intact stress contour."""
    if result['status'] not in ('completed','completed_window','failed','model_boundary','stopped'):
        raise ValueError('Render a stopped run, not an active writer')
    steps=result['steps']
    if not steps:raise ValueError('No accepted states to render')
    output.mkdir(parents=True)
    hashes={};depths=np.array([s['depth_mm'] for s in steps])
    if np.any(np.diff(depths)<=0):raise ValueError('Non-increasing accepted history')
    def load(index):
        path=source/'states'/f'{index:04d}.npz'
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:
            x=z['positions'];tri=z['triangles'];n=int(z['body_nodes']);damage=z['damage']
            if abs(float(z['depth_mm'])-depths[index])>1e-12:raise ValueError('State depth mismatch')
            if not np.isfinite(x).all() or np.any(z['J']<=0):raise ValueError('Invalid accepted geometry')
            if not np.isfinite(damage).all() or np.any((damage<0)|(damage>1)):raise ValueError('Invalid damage')
            stress=np.maximum(np.linalg.eigvalsh(z['cauchy'])[:,-1],0.)
            if not np.isfinite(stress).all():raise ValueError('Invalid saved stress')
            tip=x[n:][np.argmin(x[n:,1])]
            return x-tip,tri,n,damage,stress
    stress_max=max(.001,max(s['maximum_principal_cauchy_mpa'] for s in steps))
    def field(ax,index,kind):
        xy,tri,n,d,stress=load(index)
        if kind=='stress':
            artist=ax.tripcolor(xy[:n,0],xy[:n,1],tri,facecolors=stress,shading='flat',cmap='cividis',vmin=0,vmax=stress_max,rasterized=True)
        else:
            artist=ax.tripcolor(xy[:n,0],xy[:n,1],tri,d,shading='gouraud',cmap='magma',vmin=0,vmax=1,rasterized=True)
        ax.triplot(xy[:n,0],xy[:n,1],tri,color='#777777',lw=.1,alpha=.4)
        ax.add_patch(Polygon(xy[n:],fc='#dddddd',ec='#444444',lw=.8))
        ax.plot(0,0,'+',color='black',ms=7)
        if d.max()>1e-12:
            peak=int(d.argmax());ax.plot(*xy[peak],'*',color='white',markeredgecolor='black',ms=9)
        if result['parameters']['case']=='patch':
            ax.set(xlim=(-.1,1.5),ylim=(-1.1,.4))
        else:ax.set(xlim=(-1.3,1.3),ylim=(-1.5,.8))
        ax.set(aspect='equal',xlabel='x - blade tip (mm)',ylabel='z - blade tip (mm)',
               title=f'{kind}: depth {depths[index]:.5g} mm')
        return artist
    with plt.rc_context({'font.size':10}):
        fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
        energy_note='; energy balance unresolved' if result.get('energy_balance_resolved') is False else ''
        fig.suptitle(f'Experimental directional fracture | status: {result["status"]}\nAccepted states only; no experimental validation{energy_note}',fontsize=13)
        axes[0,0].plot(depths,[s['reaction_fz_n'] for s in steps],'-o',ms=2,lw=1)
        start = result.get('window_start_mm') if result.get('run_kind') == 'nucleation_window' else min(0.,depths[0])
        axes[0,0].set(xlabel='Prescribed depth (mm)',ylabel='Fz (N)',xlim=(start,result['target_mm']))
        axes[0,0].axvline(depths[-1],ls=':',color='#555555',label='last accepted')
        axes[0,0].legend()
        axes[0,1].plot(depths,[s['max_damage'] for s in steps],'-o',ms=2,lw=1)
        axes[0,1].set(xlabel='Prescribed depth (mm)',ylabel='Maximum damage',ylim=(0,1),xlim=(start,result['target_mm']))
        axes[0,1].axhline(result['diffuse_damage_limit'],ls='--',color='#555555',label='model boundary, not fracture calibration')
        axes[0,1].legend(fontsize=8)
        for ax in axes[0]:ax.xaxis.set_major_locator(MaxNLocator(5))
        for ax,kind,label in ((axes[1,0],'stress','Max principal Cauchy tension (MPa)'),(axes[1,1],'damage','Damage d: 0 intact, 1 fully degraded')):
            artist=field(ax,len(steps)-1,kind);fig.colorbar(artist,ax=ax,label=label,shrink=.8)
        fig.savefig(output/'summary.png',dpi=170);plt.close(fig)
        movie_status='skipped: disabled by request' if not movie else 'skipped: ffmpeg unavailable'
        if movie and shutil.which('ffmpeg'):
            # Explicit index decimation only; no interpolation and no claim that
            # fixed playback fps is experimental time under adaptive loading.
            indices=np.unique(np.r_[np.arange(0,len(steps),max(1,int(np.ceil(len(steps)/120)))),len(steps)-1]).astype(int)
            fig,axes=plt.subplots(1,2,figsize=(10,5),layout='constrained')
            writer=animation.FFMpegWriter(fps=8,metadata={'comment':'Accepted states; playback is not physical time'})
            with writer.saving(fig,str(output/'deformation.mp4'),dpi=120):
                for index in indices:
                    for ax,kind in zip(axes,('stress','damage')):
                        ax.clear();field(ax,int(index),kind)
                    fig.suptitle(f'{result["status"]} | accepted-state playback, not physical time{energy_note}\nStress scale 0–{stress_max:.4g} MPa; damage scale 0–1')
                    writer.grab_frame()
            plt.close(fig);movie_status='completed'
        else:indices=np.array([],int)
    write_json(output/'manifest.json',dict(result_sha256_at_render=hashlib.sha256((source/'result.json').read_bytes()).hexdigest(),
        states_sha256=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status=result['status'],last_accepted_depth_mm=float(depths[-1]),target_mm=result['target_mm'],
        stress_source='saved damaged constitutive stress, all 3 principal values',damage_scale=[0,1],stress_scale=[0,stress_max],
        movie_status=movie_status,movie_indices=indices.tolist(),
        transformations=['Translate by current blade tip','P1 damage interpolation','Flat element stress, no averaging','No deformation amplification','Movie index decimation; playback not physical time'],
        alt_text='Force and maximum damage over accepted loading states, followed by the last accepted local stress and damage meshes. No curves are extrapolated to the target.',
        destination='local provisional numerical review; no journal specification'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-movie',action='store_true',help='Render summary only for a bounded stopped-run review')
    args=parser.parse_args()
    if args.output.exists(): parser.error('Output must not exist')
    result=json.loads((args.source/'result.json').read_text(encoding='utf-8'))
    if result.get('framework')=='directional-rational-at1-v1':
        render_continuous(args.source,args.output,result,movie=not args.no_movie)
        return
    probes=result['probes']
    args.output.mkdir(parents=True)
    hashes={}; captions=[]
    # Use the same colour scale for every angle and panel, with explicit range.
    vmax=max(.1, max(p.get('max_damage',0.) for p in probes))
    norm=Normalize(0,vmax)
    with plt.rc_context({'font.size':10}):
        for angle in sorted({p['angle'] for p in probes}):
            selected=[p for p in probes if p['angle']==angle]
            depths=sorted({p['depth_mm'] for p in selected}); lengths=sorted({p['ell_mm'] for p in selected})
            fig,axes=plt.subplots(len(depths),len(lengths),figsize=(12,8),squeeze=False,layout='constrained')
            fig.suptitle(f'{angle:g} degrees: frozen-displacement AT1 diagnostic\n'
                         'Total elastic energy drives damage; no mechanics recoupling or crack growth',fontsize=14)
            artist=None
            for i,depth in enumerate(depths):
                for j,ell in enumerate(lengths):
                    ax=axes[i,j]
                    p=next((p for p in selected if p['depth_mm']==depth and p['ell_mm']==ell),None)
                    if p is None or not p.get('converged'):
                        ax.text(.5,.5,'Missing / failed probe',ha='center',transform=ax.transAxes);continue
                    file=args.source/f'angle{angle}_depth{depth:g}_ell{ell:g}.npz'
                    hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
                    with np.load(file) as z:
                        x=z['positions']; n=int(z['body_nodes']); tri=z['triangles'];d=z['damage']
                        if not np.isfinite(d).all() or np.any((d<0)|(d>1)): raise ValueError('Invalid damage')
                        tip=x[n:][np.argmin(x[n:,1])]; xy=x-tip
                        artist=ax.tripcolor(xy[:n,0],xy[:n,1],tri,d,shading='gouraud',cmap='cividis',norm=norm,rasterized=True)
                        ax.triplot(xy[:n,0],xy[:n,1],tri,color='#777777',lw=.12,alpha=.4)
                        ax.add_patch(Polygon(xy[n:],fc='#dddddd',ec='#555555',lw=.8))
                        peak=int(d.argmax())
                        if d.max() > 1e-12:
                            ax.plot(*xy[peak],'*',color='black',markeredgecolor='white',ms=11,mew=.7)
                        ax.plot(0,0,'+',color='black',ms=9)
                    ax.set(xlim=(-.12,.12),ylim=(-.1,.14),aspect='equal',
                           xlabel='x - blade tip (mm)',ylabel='z - blade tip (mm)',
                           title=f'Depth {depth:g} mm | ell {ell:g} mm\nmax d = {p["max_damage"]:.5f}')
            if artist is not None:
                fig.colorbar(artist,ax=axes.ravel().tolist(),label='Damage d (0 intact, 1 broken); common displayed range')
            fig.savefig(args.output/f'angle{angle}_damage.png',dpi=180)
            plt.close(fig)
            captions.append(f'{angle:g} degrees: star marks global maximum damage; plus marks blade tip. '
                            'All panels use the same current-coordinate window and damage scale. '
                            'P1 nodal interpolation; deformation is not amplified. This is not a visible crack prediction.')
    write_json(args.output/'manifest.json',dict(inputs_sha256=hashes,
        result_sha256=hashlib.sha256((args.source/'result.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        matplotlib=matplotlib.__version__,normalization=[0,vmax],dpi=180,
        panel_window_mm=[-.12,.12,-.1,.14],alt_text=captions,
        destination='provisional local research review, no journal specification',
        transformations=['Translate current coordinates by blade tip','Native P1 scalar interpolation; no data smoothing']))


if __name__=='__main__': main()
