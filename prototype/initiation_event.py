"""First strength-threshold event diagnostics, not a crack-growth law."""
import numpy as np
from .model_config import local_to_lab,_rotation_vectors


class CandidateRefinement(Exception):
    """Discard a trial and resolve the first threshold bracket more finely."""


class CandidateReached(Exception):
    """Stop only after committing and saving a threshold-crossing state."""


def criterion_values(fields,criterion,strength):
    keys={'local_nominal':'nominal_tensile_mpa','multiaxial':'multiaxial_strength_index'}
    if criterion not in keys:raise ValueError('Unknown stopping criterion')
    if keys[criterion] not in fields:raise ValueError('Requested strength criterion is unavailable')
    return np.asarray(fields[keys[criterion]])/(strength if criterion=='local_nominal' else 1.)


def location_evidence(problem,fields,depth,criterion):
    c=problem.config;ratio=criterion_values(fields,criterion,c.tensile_strength_n_mm2)
    _,en=_rotation_vectors(c.inclination_deg);r=problem.options.tip_radius_mm
    first_contact=local_to_lab(np.array([[c.drop_s_mm,c.height_mm]]),c)[0]
    tip=first_contact+r*en-np.array([0.,depth+r])
    index=int(np.argmax(ratio));point=fields['current_centroid_xz_mm'][index]
    # Include the whole superlevel set; top elements are display aids only.
    order=np.argsort(ratio)[::-1][:10]
    elements=fields.get('field_element_ids',np.arange(len(ratio)))
    return dict(criterion=criterion,kind='strength-threshold candidate; physical nucleation unvalidated',
        peak_element=int(elements[index]),peak_sample=index,peak_ratio=float(ratio[index]),blade_lowest_point_current_xz_mm=tip.tolist(),
        peak_current_xz_mm=point.tolist(),peak_reference_sn_mm=fields['reference_centroid_sn_mm'][index].tolist(),
        offset_from_blade_lowest_point_xz_mm=(point-tip).tolist(),
        supercritical_elements=np.unique(elements[ratio>=1.]).tolist(),
        top_elements=[dict(element=int(elements[i]),sample=int(i),ratio=float(ratio[i]),reference_sn_mm=fields['reference_centroid_sn_mm'][i].tolist(),
            current_xz_mm=fields['current_centroid_xz_mm'][i].tolist()) for i in order],
        localization_note=str(fields.get('field_sampling','Element centroid'))+'; mesh-dependent strength screening; offset is in the current configuration, not a measured crack origin')


def plot_location(path,problem,u,fields,evidence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xy=problem.mesh.reference_xy+u.reshape(-1,2)
    ratio=criterion_values(fields,evidence['criterion'],problem.config.tensile_strength_n_mm2)
    triangles=problem.mesh.triangles
    if 'field_element_ids' in fields:
        maximum=np.zeros(len(triangles));np.maximum.at(maximum,fields['field_element_ids'],ratio)
        triangles=problem.mesh.plot_triangles;ratio=np.repeat(maximum,4)
    point=np.array(evidence['peak_current_xz_mm']);tip=np.array(evidence['blade_lowest_point_current_xz_mm'])
    with plt.rc_context({'font.size':9}):
        fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
        for ax in axes:
            im=ax.tripcolor(xy[:,0],xy[:,1],triangles,facecolors=ratio,cmap='cividis',vmin=0,vmax=max(1.,float(ratio.max())))
            ax.plot(*point,'r*',ms=11,label='Peak strength candidate')
            ax.plot(*tip,'kx',ms=7,label='Lowest blade point')
            ax.set(xlabel='Current laboratory x (mm)',ylabel='Current laboratory z (mm)',aspect='equal')
        axes[0].set_title('Whole specimen: strength screening, not a crack field')
        center=(point+tip)/2;half=max(.3,float(np.max(np.abs(point-tip)))/2+.15)
        axes[1].set(xlim=(center[0]-half,center[0]+half),ylim=(center[1]-half,center[1]+half),title='Candidate and blade offset')
        axes[1].legend(loc='best');fig.colorbar(im,ax=axes,label='Strength ratio')
        fig.savefig(path,dpi=150);plt.close(fig)
