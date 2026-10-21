"""Read-only common-material sensitivity/identifiability analysis, not FE fit."""
import argparse
import copy
from html import escape
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.optimize import linprog,lsq_linear
from tools import run_common_material_study as host
from prototype.history_sensitivity import PARAMETERS
from prototype.material_case import load_force_csv
from tools.render_solver_dashboard import atomic_write,shell,stamp
from run_support import write_json

ROOT=Path(__file__).resolve().parents[1]
BOUNDS=np.array([.1,.3,.2])


def prefix(case):
    rows=[]
    for row in case.get('steps',[]):
        if not row['passed']:break
        rows.append(row)
    return rows


def descent(gradients):
    g=np.asarray(gradients)*BOUNDS
    norm=np.linalg.norm(g,axis=1)
    if np.any(norm<1e-14):return dict(status='stationary_objective_present',strict_common_descent=False)
    rows=g/norm[:,None]
    fit=linprog([0.,0.,0.,-1.],A_ub=np.column_stack((rows,np.ones(len(rows)))),b_ub=np.zeros(len(rows)),
                bounds=[(-1.,1.)]*3+[(None,None)],method='highs')
    if not fit.success:raise RuntimeError(fit.message)
    d=fit.x[:3]*BOUNDS
    result=dict(status='completed',strict_common_descent=bool(fit.x[3]>1e-7),normalized_margin=float(fit.x[3]),
                coordinate_direction=d.tolist(),objective_directional_derivatives=(np.asarray(gradients)@d).tolist(),
                meaning='Infinitesimal common direction; its finite endpoint has not been recomputed by FE')
    if not result['strict_common_descent']:
        certificate=linprog(np.zeros(len(rows)),A_eq=np.vstack((rows.T,np.ones(len(rows)))),
                            b_eq=[0.,0.,0.,1.],bounds=(0,None),method='highs')
        if certificate.success:
            residual=float(np.linalg.norm(rows.T@certificate.x))
            result['linearized_conflict_certificate']=dict(weights=certificate.x.tolist(),
                weighted_normalized_gradient_norm=residual,sum_weights=float(sum(certificate.x)),
                verified=bool(residual<1e-8 and min(certificate.x)>=-1e-10 and abs(sum(certificate.x)-1)<1e-8),
                meaning='Nonnegative unit-sum gradients cancel: all cannot have a strictly negative dot product with one direction')
    return result


def common_step(curves,direction,*,include_slope_change=False):
    """A bounded step along a common descent ray, only in the tangent model.

    Limit to the minimum of each individual quadratic's minimizer. Thus an
    aggregate improvement cannot conceal a worse angle or increment objective.
    """
    direction=np.asarray(direction);fractions=[1.];features=[]
    for c in curves.values():
        error=np.asarray(c['baseline_fz_n'])-c['experimental_fz_n']
        response=np.asarray(c['derivatives_n'])@direction
        widths=np.diff(c['slope_points_mm'])
        shape=(np.asarray(c['baseline_slopes_n_mm'])-c['experimental_slopes_n_mm'])*widths
        shape_response=(np.asarray(c['slope_derivatives'])@direction)*widths
        objectives=[(error,response),(shape,shape_response)]
        if include_slope_change:
            objectives.append((np.diff(np.asarray(c['baseline_slopes_n_mm'])-c['experimental_slopes_n_mm']),
                               np.diff(np.asarray(c['slope_derivatives'])@direction)))
        for e,v in objectives:
            slope=float(e@v);curvature=float(v@v)
            if slope>=0 or curvature<=0:
                return dict(status='no_strict_common_step',fraction=0.)
            fractions.append(-slope/curvature);features.append((e,v))
    fraction=float(min(fractions));delta=fraction*direction;rows={}
    for angle,c in curves.items():
        predicted=np.asarray(c['baseline_fz_n'])+np.asarray(c['derivatives_n'])@delta
        slopes=np.asarray(c['baseline_slopes_n_mm'])+np.asarray(c['slope_derivatives'])@delta
        rows[angle]=dict(rmse_n=float(np.sqrt(np.mean((predicted-c['experimental_fz_n'])**2))),
                         slopes_n_mm=slopes.tolist())
    return dict(status='completed',fraction=fraction,coordinate_change=delta.tolist(),curves=rows,
                objective_ratios=[float(np.sum((e+fraction*v)**2)/np.sum(e**2)) for e,v in features],
                scope='Tangent model only; each requested quadratic error decreases, not a nonlinear FE guarantee')


def joint_analysis(cases,end,*,split=None):
    grid=np.arange(10,int(round(end*100))+1)/100
    points=np.array([.1,.25,end]) if end>.25 else np.array([.1,(.1+end)/2,end])
    if split is not None:
        if not .1<split<end:raise ValueError('Secant split must be strictly inside the common window')
        points=np.array([.1,split,end])
    if end<.15-1e-10:return dict(status='insufficient_common_window',end_mm=end)
    errors=[];jac=[];grad_force=[];grad_shape=[];grad_change=[];curves={};inputs={}
    for angle,case in cases.items():
        rows=prefix(case);d=np.array([r['depth_mm'] for r in rows]);f=np.array([r['fz_n'] for r in rows])
        derivative=np.array([[r['parameters'][p]['derivative_n'] for p in PARAMETERS] for r in rows])
        path=ROOT/'docs'/host.EXPERIMENTS[angle];ed,ef,rate=load_force_csv(path);inputs[str(path)]=host.sha(path)
        if abs(rate-.2)>1e-12:raise ValueError('Experimental speed mismatch')
        v=np.interp(grid,d,f);target=np.interp(grid,ed,ef);e=v-target
        J=np.column_stack([np.interp(grid,d,derivative[:,i]) for i in range(3)])
        vpoints=np.interp(points,d,f);epoints=np.interp(points,ed,ef)
        dp=np.column_stack([np.interp(points,d,derivative[:,i]) for i in range(3)])
        es=np.diff(vpoints-epoints);Js=np.diff(dp,axis=0)
        errors.extend(e/np.sqrt(3*len(grid)));errors.extend(.5*es/np.sqrt(6))
        jac.extend(J/np.sqrt(3*len(grid)));jac.extend(.5*Js/np.sqrt(6))
        grad_force.append(2*J.T@e/len(e));grad_shape.append(2*Js.T@es/len(es))
        slopes=np.diff(vpoints)/np.diff(points);experimental_slopes=np.diff(epoints)/np.diff(points)
        slope_derivatives=Js/np.diff(points)[:,None]
        change_error=float(np.diff(slopes-experimental_slopes)[0]);change_derivative=np.diff(slope_derivatives,axis=0)[0]
        grad_change.append(2*change_error*change_derivative)
        curves[str(angle)]=dict(depth_mm=grid.tolist(),baseline_fz_n=v.tolist(),experimental_fz_n=target.tolist(),
            derivatives_n=J.tolist(),slope_points_mm=points.tolist(),baseline_slopes_n_mm=(np.diff(vpoints)/np.diff(points)).tolist(),
            experimental_slopes_n_mm=(np.diff(epoints)/np.diff(points)).tolist(),slope_derivatives=(Js/np.diff(points)[:,None]).tolist(),
            baseline_rmse_n=float(np.sqrt(np.mean(e**2))),
            secant_slope_change_n_mm=dict(baseline=float(np.diff(slopes)[0]),experiment=float(np.diff(experimental_slopes)[0]),
                derivatives=change_derivative.tolist(),error=change_error))
    J=np.array(jac);e=np.array(errors);singular=np.linalg.svd(J,compute_uv=False)
    unit=J/np.linalg.norm(J,axis=0);normalized_singular=np.linalg.svd(unit,compute_uv=False)
    fit=lsq_linear(J,-e,bounds=(-BOUNDS,BOUNDS),tol=1e-12,max_iter=200)
    if not fit.success:raise RuntimeError(fit.message)
    for angle,c in curves.items():
        prediction=np.array(c['baseline_fz_n'])+np.array(c['derivatives_n'])@fit.x
        c['linearized_fz_n']=prediction.tolist()
        c['linearized_rmse_n']=float(np.sqrt(np.mean((prediction-np.array(c['experimental_fz_n']))**2)))
        c['linearized_slopes_n_mm']=(np.array(c['baseline_slopes_n_mm'])+np.array(c['slope_derivatives'])@fit.x).tolist()
    q0=host.BASE[0]/sum(host.BASE[:2]);g0=sum(host.BASE[:2]);theta=fit.x
    q=1/(1+np.exp(-(np.log(q0/(1-q0))+theta[1])))
    common=descent(grad_force+grad_shape)
    common_projection=(common_step(curves,common['coordinate_direction']) if common.get('strict_common_descent')
                       else dict(status='no_common_descent_direction'))
    complete=descent(grad_force+grad_shape+grad_change)
    complete_projection=(common_step(curves,complete['coordinate_direction'],include_slope_change=True)
                         if complete.get('strict_common_descent') else dict(status='no_common_descent_direction'))
    return dict(status='completed',window_mm=[.1,float(end)],full_planned_window=bool(end>=.5-1e-10),
        coordinates=list(PARAMETERS),bounds=BOUNDS.tolist(),source_sha256=inputs,curves=curves,
        singular_values=singular.tolist(),column_normalized_singular_values=normalized_singular.tolist(),
        column_cosines=(unit.T@unit).tolist(),condition_number=float(singular[0]/singular[-1]),
        normalized_condition_number=float(normalized_singular[0]/normalized_singular[-1]),
        bounded_linear_projection=dict(coordinate_change=theta.tolist(),cost_before=float(e@e),cost_after=float(np.linalg.norm(J@theta+e)**2),
            illustrative_parameters=dict(g0_mpa=float(g0*np.exp(theta[0])),q=float(q),tau_s=float(host.BASE[2]*np.exp(theta[2]))),
            scope='Only tangent-space projection, not identified parameters, FE prediction, or automatic loading proposal'),
        common_force_descent=descent(grad_force),common_force_and_shape_descent=common,
        common_tangent_step=common_projection,
        common_force_shape_and_slope_change_descent=complete,complete_common_tangent_step=complete_projection,
        nine_objective_order=['force_0','force_33','force_57','increments_0','increments_33','increments_57','slope_change_0','slope_change_33','slope_change_57'],
        objective_definition='Equal-angle mean squared Fz errors plus .25 times squared two-increment errors; exported zeros unchanged',
        disclaimer='No statistical confidence intervals or global identifiability claim; early intact window provisional')


def report(out,final=False):
    manifest=host.read(out/'manifest.json');queue=host.read(out/'queue.json') if (out/'queue.json').exists() else {'status':'prepared'}
    cases={a:host.read(out/f'angle{a}/result.json') for a in (0,33,57) if (out/f'angle{a}/result.json').exists()}
    body='<h1>共同材料三参数：完整历史联合检查</h1><p>'+stamp()+' · 队列 '+queue['status']+'</p>'
    body+='<p>解析摩擦历史偏导；原生物理方程、K、μ、几何、边界及实验零点不变。没有新FE加载。两个原失败夹具已重新解方向并通过预检；回归数量和结果见测试日志。</p>'
    body+='<p><a href="plan.md">冻结方案</a> · <a href="preflight.json">两场预检</a> · <a href="regressions.log">启动前测试</a> · <a href="joint_regressions.log">60项联合回归</a> · <a href="report_regressions.log">共同方向及冲突证据复核</a> · <a href="queue.json">进程记录</a></p>'
    body+='<table><tr><th>角度</th><th>诊断状态</th><th>连续有效前缀 mm</th><th>状态数</th><th>实际日志</th></tr>'
    for angle in (0,33,57):
        case=cases.get(angle,{});rows=prefix(case)
        label='诊断预算结束' if 'TimeoutError' in case.get('error','') else case.get('status','pending')
        body+=f'<tr><td>{angle}°</td><td>{label}</td><td>{rows[-1]["depth_mm"] if rows else "无"}</td><td>{len(rows)}</td><td><a href="angle{angle}/index.html">HTML日志</a></td></tr>'
        if case.get('error'):body+='<tr><td colspan="5">'+escape(case['error'])+'。诊断终止不等于原加载失效。</td></tr>'
    body+='</table>'
    preflight=host.read(out/'preflight.json')
    body+='<h2>解析历史偏导的两场预检</h2><p>33°/.17 mm与57°/零点都重新求解修正方向，再做真实扰动。此表采用旧上一导数作为条件夹具；下文的正式诊断则从rest重新传播。</p><table><tr><th>角度/场编号</th><th>扰动尺度</th><th>自由残差余项相对值</th><th>反力导数相对误差</th></tr>'
    for f in preflight['fields']:
        for c in f['parameters']['log_tau']['checks']:
            body+=f'<tr><td>{f["angle"]}° / {f["index"]}</td><td>{c["eps"]}</td><td>{c.get("residual_relative")}</td><td>{c.get("force_relative")}</td></tr>'
    body+='</table><p>诊断门槛均为2%；57°采用预设尺度序列中的可分辨扰动，没有放宽物理平衡验收。</p>'
    if final:
        if queue['status']!='ended' or len(cases)!=3 or any(c['status']=='running' for c in cases.values()):raise RuntimeError('Not a final completed diagnostic record')
        verified={'source':0,'input':0}
        for name,h in manifest['source_sha256'].items():
            if host.sha(out/'source'/name)!=h:raise ValueError('Frozen source changed '+name)
            verified['source']+=1
        inputs={k:v for c in cases.values() for k,v in c['inputs'].items()}
        inputs.update(host.read(out/'preflight.json')['inputs'])
        for name,h in inputs.items():
            if host.sha(Path(name))!=h:raise ValueError('Input changed '+name)
            verified['input']+=1
        additional=host.read(out/'additional_test_sources.json')
        for name,h in additional.items():
            if name=='runtime_manifest_unchanged':continue
            if host.sha(out/'verification'/Path(name).name)!=h:raise ValueError('Additional test source changed '+name)
        verified['additional_test_sources']=len(additional)-1
        end=min(prefix(c)[-1]['depth_mm'] if prefix(c) else -1 for c in cases.values())
        joint=joint_analysis(cases,end);joint['verified']=verified
        joint['execution']={str(a):dict(status=c['status'],error=c.get('error'),elapsed_s=c.get('elapsed_s'),
            qualified_states=len(prefix(c)),linear_calls=len(c.get('linear_calls',[])),
            linear_fallbacks=sum(r.get('fallback_reason') is not None for r in c.get('linear_calls',[])),
            linear_s=sum(r['seconds'] for r in c.get('linear_calls',[])),
            recorded_step_s=sum(r['elapsed_s'] for r in c['steps'])) for a,c in cases.items()}
        if joint['status']=='completed':
            second=copy.deepcopy(cases)
            for case in second.values():
                for row in case['steps']:
                    for p in PARAMETERS:row['parameters'][p]['derivative_n']=row['parameters'][p]['derivatives_n'][0]
            other=joint_analysis(second,end)
            joint['independent_chain_comparison']=dict(
                normalized_condition_numbers=[joint['normalized_condition_number'],other['normalized_condition_number']],
                common_six_objective_margins=[joint['common_force_and_shape_descent'].get('normalized_margin'),other['common_force_and_shape_descent'].get('normalized_margin')],
                common_nine_objective_margins=[joint['common_force_shape_and_slope_change_descent'].get('normalized_margin'),other['common_force_shape_and_slope_change_descent'].get('normalized_margin')],
                bounded_projection_coordinate_difference=(np.array(joint['bounded_linear_projection']['coordinate_change'])-other['bounded_linear_projection']['coordinate_change']).tolist())
            windows=[]
            for endpoint in sorted(set([w for w in (.2,.25,.3,.4,.5) if w<=end+1e-10]+[float(end)])):
                r=joint_analysis(cases,endpoint)
                balanced=joint_analysis(cases,endpoint,split=(.1+endpoint)/2)
                windows.append(dict(end_mm=endpoint,slope_points_mm=r['curves']['0']['slope_points_mm'],
                    normalized_condition_number=r['normalized_condition_number'],
                    six_objective_margin=r['common_force_and_shape_descent'].get('normalized_margin'),
                    nine_objective_margin=r['common_force_shape_and_slope_change_descent'].get('normalized_margin'),
                    nine_objective_certificate=r['common_force_shape_and_slope_change_descent'].get('linearized_conflict_certificate'),
                    balanced_split_nine_objective_margin=balanced['common_force_shape_and_slope_change_descent'].get('normalized_margin')))
            joint['window_diagnostics']=windows
        write_json(out/'joint_analysis.json',joint)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        with plt.rc_context({'font.size':10,'axes.grid':True,'grid.alpha':.2}):
            fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
            for angle,color,style in zip((0,33,57),('#0072B2','#D55E00','#009E73'),('-','--','-.')):
                rows=prefix(cases[angle]);d=[r['depth_mm'] for r in rows]
                for ax,key in zip(axes,PARAMETERS):ax.plot(d,[r['parameters'][key]['derivative_n'] for r in rows],color=color,ls=style,label=f'{angle}°')
            for ax,title in zip(axes,('log(G0)','logit(q)','log(tau)')):ax.set(xlabel='Vertical depth (mm)',ylabel='Fz derivative (N)',title=title,xlim=(0,.5));ax.legend()
            fig.savefig(out/'sensitivities.png',dpi=170,facecolor='white');fig.savefig(out/'sensitivities.svg',facecolor='white');plt.close(fig)
            if joint['status']=='completed':
                fig,axes=plt.subplots(2,3,figsize=(12,6.3),layout='constrained',sharex=True,sharey='row')
                common=joint['complete_common_tangent_step']
                common_label='Nine-objective tangent step'
                if common['status']!='completed':
                    common=joint['common_tangent_step'];common_label='Six-objective tangent step'
                for col,(angle,c) in enumerate(joint['curves'].items()):
                    d=np.array(c['depth_mm']);target=np.array(c['experimental_fz_n'])
                    baseline=np.array(c['baseline_fz_n']);local=np.array(c['linearized_fz_n'])
                    upper,lower=axes[:,col]
                    upper.plot(d,target,color='#222222',lw=2,label='Experiment')
                    for value,color,style,label in ((baseline,'#0072B2','-','Saved FE baseline'),
                        (local,'#D55E00','--','Total-error tangent projection')):
                        upper.plot(d,value,color=color,ls=style,label=label)
                        lower.plot(d,value-target,color=color,ls=style)
                    if common['status']=='completed':
                        value=baseline+np.array(c['derivatives_n'])@common['coordinate_change']
                        upper.plot(d,value,color='#009E73',ls=':',lw=2,label=common_label)
                        lower.plot(d,value-target,color='#009E73',ls=':',lw=2)
                    upper.set(title=f'{angle}°',ylabel='Fz (N)');lower.set(xlabel='Vertical depth (mm)',ylabel='Fz - experiment (N)')
                    lower.axhline(0,color='#666666',lw=.7)
                axes[0,0].legend(fontsize=8)
                fig.suptitle('Local diagnostic projections are NOT new nonlinear FE solutions',fontsize=11)
                fig.savefig(out/'joint_comparison.png',dpi=170,facecolor='white');fig.savefig(out/'joint_comparison.svg',facecolor='white');plt.close(fig)
        body+='<h2>连续合格的历史导数</h2><p>曲线只画连续合格前缀；没有平滑、外推或缺失补零。参数坐标不同，各面板的导数量级不可直接解释成材料重要性排名。</p><img style="width:100%" src="sensitivities.png" alt="三个共同参数下，各角度完整历史反力敏感性；只显示通过资格的前缀。">'
        body+='<p>冻结源/输入哈希核对：'+escape(str(verified))+'。</p>'
        if joint['status']=='completed':
            body+='<h2>三角度共同窗口的局部联合判断</h2><p>窗口 '+str(joint['window_mm'])+' mm；完整预定.5 mm窗口：'+str(joint['full_planned_window'])+'。</p>'
            if not joint['full_planned_window']:
                body+='<p><strong>计划的完整0.5 mm共同窗口尚未取得。下列判断只使用实际合格的较早窗口，不能替代完整窗口验证或据此产生加载参数提案。</strong></p>'
            body+=('<p>归一化Jacobian条件数 '+f'{joint["normalized_condition_number"]:.3g}'+
                '。存在三个角度力值误差的共同一阶下降方向：'+str(joint['common_force_descent'].get('strict_common_descent'))+
                '；同时要求各角度力值与两段增量误差下降：'+str(joint['common_force_and_shape_descent'].get('strict_common_descent'))+'。</p>')
            body+='<p>长期份额与松弛时间的导数列夹角余弦为 '+f'{joint["column_cosines"][1][2]:.6f}'+'，局部响应方向高度相似；这支持参数耦合强，但不是统计置信区间或全局不可识别证明。两条独立差分链给出一致的联合判断。</p>'
            body+='<img style="width:100%" src="joint_comparison.png" alt="三个角度各自实验、原保存FE曲线及两个局部线性化投影；下排为对实验的力误差。投影不是新FE加载结果。"><p>实验数值及导出零点保持；只有蓝线是已完成的FE路径。橙线与绿线是局部诊断，不用于宣称复现完成。</p>'
            body+='<p>下表仅为有界切线空间投影，未重新计算平衡，不是新仿真或标定参数。有限变动可能跨接触分支，不能据此承诺误差下降。</p>'
            body+='<table><tr><th>角度</th><th>基线RMSE N</th><th>线性化RMSE N</th><th>基线两段斜率 N/mm</th><th>线性化斜率 N/mm</th><th>实验斜率 N/mm</th></tr>'
            for angle,c in joint['curves'].items():
                values=[angle,c['baseline_rmse_n'],c['linearized_rmse_n'],c['baseline_slopes_n_mm'],c['linearized_slopes_n_mm'],c['experimental_slopes_n_mm']]
                body+='<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in values)+'</tr>'
            body+='</table><p>坐标变化边界 '+str(BOUNDS.tolist())+'；投影坐标变化 '+str(joint['bounded_linear_projection']['coordinate_change'])+'。边界与统计噪声无关，只限定局部检验，不能用未通过判全参数族无能力。</p>'
            step=joint['common_tangent_step']
            body+='<h3>逐角度保护的切线方向</h3><p>上表只最小化总误差，允许牺牲某一角度。此处单独检查六个目标：每个角度的力值误差及两段增量误差均下降。仍只是在切线模型中，不是新的加载结果。</p>'
            if step['status']=='completed':
                body+='<p>沿共同方向取系数 '+f'{step["fraction"]:.4g}'+'，坐标变化 '+str(step['coordinate_change'])+'。</p><table><tr><th>角度</th><th>线性化RMSE N</th><th>线性化两段斜率 N/mm</th></tr>'
                for angle,c in step['curves'].items():body+='<tr><td>'+angle+'</td><td>'+f'{c["rmse_n"]:.6g}'+'</td><td>'+escape(str(c['slopes_n_mm']))+'</td></tr>'
                body+='</table><p>六项目标的新/旧误差平方比：'+escape(str(step['objective_ratios']))+'。比值接近1表示改善很小；这类结果不支持大幅调参承诺。</p>'
            else:body+='<p>未取得满足六目标共同下降的局部方向。该结果仅适用于当前点、窗口和所选三个参数。</p>'
            body+='<h3>单独约束斜率演化</h3><p>两段增量的总误差减小仍可能掩盖斜率演化变差。因此额外检查“后段割线斜率−前段割线斜率”，并要求三个角度该误差也下降，共九个目标。这是有限窗口的形状指标，不是瞬时二阶导数。</p><table><tr><th>角度</th><th>原FE斜率差 N/mm</th><th>实验斜率差 N/mm</th></tr>'
            for angle,c in joint['curves'].items():
                v=c['secant_slope_change_n_mm'];body+=f'<tr><td>{angle}°</td><td>{v["baseline"]:.6g}</td><td>{v["experiment"]:.6g}</td></tr>'
            body+='</table><p>九目标共同一阶下降方向存在：'+str(joint['common_force_shape_and_slope_change_descent'].get('strict_common_descent'))+'。六目标下降不自动代表这一更完整的形状检查通过。</p>'
            complete=joint['complete_common_tangent_step']
            if complete['status']=='completed':
                body+='<table><tr><th>角度</th><th>原FE RMSE N</th><th>九目标共同切线步RMSE N</th><th>预计相对改善</th></tr>'
                for angle,c in complete['curves'].items():
                    base=joint['curves'][angle]['baseline_rmse_n']
                    body+=f'<tr><td>{angle}°</td><td>{base:.6f}</td><td>{c["rmse_n"]:.6f}</td><td>{100*(1-c["rmse_n"]/base):.3f}%</td></tr>'
                body+='</table><p>该共同方向的局部改善幅度很小，不能据此承诺已找到解决三角度偏差的新参数。</p>'
            body+='<h3>窗口与分段敏感性</h3><p>以下均用同一合格历史导数，未重新加载。接近零的共同下降裕度表示没有严格共同下降方向；平衡分段把窗口分为等长两段，用于检查结论是否依赖单一分段。短区间割线对实验局部波动敏感。</p><table><tr><th>结束深度 mm</th><th>原分段 mm</th><th>列归一化条件数</th><th>六目标裕度</th><th>九目标裕度</th><th>等长分段九目标裕度</th></tr>'
            for w in joint['window_diagnostics']:
                body+='<tr>'+''.join('<td>'+escape(str(w[k]))+'</td>' for k in ('end_mm','slope_points_mm','normalized_condition_number','six_objective_margin','nine_objective_margin','balanced_split_nine_objective_margin'))+'</tr>'
            body+='</table>'
            body+='<details><summary>奇异值、列相关性与局部方向</summary><pre>'+escape(json.dumps({k:v for k,v in joint.items() if k not in ('curves','source_sha256')},ensure_ascii=False,indent=2))+'</pre></details>'
        else:body+='<p>共同合格窗口不足，未进行联合参数投影。</p>'
        body+='<p><a href="joint_analysis.json">全部联合数值、目标与边界</a></p>'
        shutil.copy2(__file__,out/'report_code.py')
        write_json(out/'report_provenance.json',dict(generated=stamp(),data_sha256={str(out/f'angle{a}/result.json'):host.sha(out/f'angle{a}/result.json') for a in cases},
            report_code_sha256=host.sha(out/'report_code.py'),figures=dict(sensitivities='12 x 3.8 in',joint_comparison='12 x 6.3 in'),
            figure_policy='170 dpi; no smoothing or confidence bands; deterministic saved paths and explicitly labeled local linearizations',matplotlib_version=matplotlib.__version__,
            source_manifest_sha256=host.sha(out/'manifest.json')))
    else:body+='<p>这是当前启动快照，不能仅凭旧running状态判定进程仍存活。各角度HTML记录独立更新。</p>'
    atomic_write(out/'index.html',shell('共同参数全历史联合检验',body))
    print('REPORT '+str(out/'index.html'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--final',action='store_true')
    a=p.parse_args();report(a.study.resolve(),a.final)
