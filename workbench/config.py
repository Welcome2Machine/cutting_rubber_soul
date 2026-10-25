"""Versioned workbench inputs, validation and translation to the maintained CLI."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'softcut-config/1'
MESH_PRESETS = {
    'standard': dict(local_spacing_mm=.05, surface_spacing_mm=.005, fine_halfwidth_mm=3., coarse_spacing_mm=1.),
    'coarse': dict(local_spacing_mm=.1, surface_spacing_mm=.01, fine_halfwidth_mm=3., coarse_spacing_mm=1.),
    'fine': dict(local_spacing_mm=.025, surface_spacing_mm=.0025, fine_halfwidth_mm=3., coarse_spacing_mm=1.),
}


def catalog():
    materials = json.loads((ROOT/'configs/materials.json').read_text(encoding='utf-8-sig'))
    blades = json.loads((ROOT/'configs/blades.json').read_text(encoding='utf-8-sig'))
    default = dict(schema_version=SCHEMA, name='橡胶斜向压入', material=materials[0], blade=blades[0],
        geometry=dict(angle_deg=57., tip_radius_mm=.02), contact=dict(condition='dry', mu=.35),
        simulation=dict(depth_mm=2., increment_mm=.01, speed_mm_s=.2, mesh='standard',
            **MESH_PRESETS['standard'], nose_facets=64, blade_spacing_mm=.005),
        resources=dict(solve_seconds=7200., total_seconds=9000., ipc_threads=2,
            linear_threads=4, linear_backend='native', material_batch_size=0),
        output=dict(max_frames=11, movie=False))
    return dict(materials=materials, blades=blades, defaults=default, mesh_presets=MESH_PRESETS,
        specimen=dict(length_mm=30., height_mm=5., width_mm=20., bottom='fixed', kinematics='plane_strain'),
        limits=dict(max_depth_mm=2., max_angle_deg=75., speed_mm_s=.2),
        supported_blade_shapes=['symmetric_wedge'], tip_shape='circular_nose')


def number(obj, key, low, high, *, exclusive_low=False, integer=False):
    value = obj.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{key} 必须是有限数值')
    if (value <= low if exclusive_low else value < low) or value > high:
        raise ValueError(f'{key} 必须在 {"(" if exclusive_low else "["}{low}, {high}] 范围内')
    if integer and (not isinstance(value, int)):
        raise ValueError(f'{key} 必须是整数')
    return value


def keys(obj, required, optional=()):
    if not isinstance(obj, dict): raise ValueError('配置段必须是 JSON 对象')
    missing = set(required)-obj.keys()
    unknown = obj.keys()-set(required)-set(optional)
    if missing or unknown: raise ValueError(f'配置字段不符；缺少 {sorted(missing)}，未知 {sorted(unknown)}')


def text_field(obj, key, maximum=120):
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip() or len(value)>maximum or any(ord(c)<32 for c in value):
        raise ValueError(f'{key} 必须是1–{maximum}个可显示字符')


def normalize_config(raw):
    """Fail closed on unsupported physics; never silently substitute controls."""
    c = deepcopy(raw)
    keys(c, ['schema_version','name','material','blade','geometry','contact','simulation','resources','output'])
    if c['schema_version'] != SCHEMA: raise ValueError(f'仅支持 {SCHEMA}')
    text_field(c,'name')
    m,b,g,t,s,r,o = (c[k] for k in ('material','blade','geometry','contact','simulation','resources','output'))
    keys(m,['id','label','law','Ginf_MPa','Gv_MPa','tau_s','K_MPa'],['source_note'])
    keys(b,['id','label','shape','thickness_mm','included_angle_deg'],['source_note'])
    for item in (m,b):
        text_field(item,'id',80); text_field(item,'label')
        if 'source_note' in item: text_field(item,'source_note',1000)
    if m['law'] != 'isochoric_maxwell': raise ValueError('仅支持已维护的等容有限应变 Maxwell 材料')
    for k in ('Ginf_MPa','Gv_MPa'): number(m,k,0.,100.,exclusive_low=True)
    number(m,'tau_s',.001,10000.); number(m,'K_MPa',0.,100000.,exclusive_low=True)
    if m['K_MPa']<=2*m['Ginf_MPa']/3: raise ValueError('当前宿主要求正泊松比：K > 2 G∞ / 3')
    if b['shape']!='symmetric_wedge': raise ValueError('当前仅支持对称双刃楔刀；不能用任意轮廓冒充已实现刀具')
    number(b,'thickness_mm',.005,5.); number(b,'included_angle_deg',2.,120.)
    keys(g,['angle_deg','tip_radius_mm']); number(g,'angle_deg',0.,75.)
    number(g,'tip_radius_mm',0.,2.,exclusive_low=True)
    if g['tip_radius_mm']>=b['thickness_mm']/2: raise ValueError('圆鼻半径必须小于刀厚的一半')
    if g['angle_deg']+b['included_angle_deg']/2>=90: raise ValueError('当前圆鼻首触构造要求倾角 + 半楔角 < 90°')
    keys(t,['condition','mu'])
    if t['condition'] not in ('dry','oil','custom'): raise ValueError('接触条件必须为 dry / oil / custom')
    number(t,'mu',0.,2.,exclusive_low=True)
    keys(s,['depth_mm','increment_mm','speed_mm_s','mesh','local_spacing_mm','surface_spacing_mm',
            'fine_halfwidth_mm','coarse_spacing_mm','nose_facets','blade_spacing_mm'])
    number(s,'depth_mm',.001,2.); number(s,'increment_mm',.000625,.05)
    if s['increment_mm']>s['depth_mm']: raise ValueError('名义物理步长不能大于总行程')
    if isinstance(s['speed_mm_s'],bool) or s['speed_mm_s']!=.2: raise ValueError('当前研究速度固定为0.2 mm/s')
    if s['mesh'] not in (*MESH_PRESETS,'custom'): raise ValueError('未知网格预设')
    number(s,'local_spacing_mm',.01,1.); number(s,'surface_spacing_mm',.001,.1)
    number(s,'fine_halfwidth_mm',.1,7.); number(s,'coarse_spacing_mm',.1,3.)
    if not s['surface_spacing_mm']<=s['local_spacing_mm']<=s['coarse_spacing_mm']:
        raise ValueError('网格间距应满足表面 ≤ 局部 ≤ 远场')
    if s['mesh'] in MESH_PRESETS and any(s[k]!=v for k,v in MESH_PRESETS[s['mesh']].items()):
        raise ValueError('网格数值与预设不一致；手动改动后请选择 custom')
    number(s,'nose_facets',8,256,integer=True)
    if s['nose_facets']%2: raise ValueError('圆鼻分段数必须为偶数')
    if s['blade_spacing_mm'] is not None: number(s,'blade_spacing_mm',.001,.05)
    keys(r,['solve_seconds','total_seconds','ipc_threads','linear_threads','linear_backend','material_batch_size'])
    number(r,'solve_seconds',5.,86400.); number(r,'total_seconds',10.,100800.)
    if r['total_seconds']<=r['solve_seconds']: raise ValueError('总墙钟预算须大于求解预算，为准备/独立审查留出时间')
    number(r,'ipc_threads',1,64,integer=True); number(r,'linear_threads',1,64,integer=True)
    number(r,'material_batch_size',0,65536,integer=True)
    if r['linear_backend'] not in ('native','pardiso'): raise ValueError('线性后端必须为 native / pardiso')
    keys(o,['max_frames','movie']); number(o,'max_frames',2,41,integer=True)
    if type(o['movie']) is not bool: raise ValueError('movie 必须为布尔值')
    return c


def describe(c):
    m,g,s,t=c['material'],c['geometry'],c['simulation'],c['contact']
    messages=['完整体平面应变：没有裂纹、损伤或油膜流体方程。',
              '新几何/网格/参数组合须独立验证；计算完成不等于实验标定。']
    if t['condition']=='oil': messages.append('油润滑仅作为给定 μ 的库仑摩擦近似；没有自动标定油的系数。')
    if s['surface_spacing_mm']>g['tip_radius_mm']/2:
        messages.append('表面间距较大，刀尖局部力与场需更细网格对照；不自动修改几何或网格。')
    if s['mesh']!='standard': messages.append('该网格为对照离散，尚无完整共同工况精度认证。')
    if c['resources']['material_batch_size']:
        messages.append('分批材料积分曾触发严格历史重现差异；最终独立审查仍保留原门槛。')
    return dict(speed_mm_s=.2, material_G0_MPa=m['Ginf_MPa']+m['Gv_MPa'],
        material_q=m['Ginf_MPa']/(m['Ginf_MPa']+m['Gv_MPa']),
        physical_duration_s=s['depth_mm']/.2, physical_dt_s=s['increment_mm']/.2,
        requested_output_states=math.ceil(s['depth_mm']/s['increment_mm']-1e-12)+1,
        force_tolerance_N=1e-4, minimum_increment_mm=.000625, messages=messages)


def solver_command(c, output, python=None):
    """Argument vector only: no shell and no commands taken from user text."""
    c=normalize_config(c);m,b,g,t,s,r=(c[k] for k in ('material','blade','geometry','contact','simulation','resources'))
    options={'output':str(Path(output).resolve()),'equilibrium-shear':m['Ginf_MPa'],'maxwell-shear':m['Gv_MPa'],
        'relaxation-s':m['tau_s'],'bulk-modulus':m['K_MPa'],'loading-rate':.2,'angle':g['angle_deg'],
        'mu':t['mu'],'blade-thickness':b['thickness_mm'],'blade-angle':b['included_angle_deg'],
        'tip-radius':g['tip_radius_mm'],'depth':s['depth_mm'],'increment':s['increment_mm'],
        'local-spacing':s['local_spacing_mm'],'surface-spacing':s['surface_spacing_mm'],
        'fine-halfwidth':s['fine_halfwidth_mm'],'coarse-spacing':s['coarse_spacing_mm'],'nose-facets':s['nose_facets'],
        'seconds':r['solve_seconds'],'threads':r['ipc_threads'],'linear-backend':r['linear_backend'],
        'linear-threads':r['linear_threads'],'material-batch-size':r['material_batch_size']}
    if s['blade_spacing_mm'] is not None: options['blade-spacing']=s['blade_spacing_mm']
    argv=[python or sys.executable,'-X','utf8','-u','-m','tools.run_material_candidate']
    for k,v in options.items(): argv.extend(['--'+k,str(v)])
    return argv
