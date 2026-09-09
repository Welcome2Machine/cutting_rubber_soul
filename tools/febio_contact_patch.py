"""Guarded, opt-in FEBio 4.13 contact diagnostics and tangent correction.

Upstream MIT source and license remain in the isolated build tree. Changes are
recorded by the build manifest; the original reference binary is not replaced.
"""
from pathlib import Path
import hashlib


TANGENT_CHECK = r'''
	// Fixed planar rigid facet, in-plane slave block only. Finite differences
	// call the actual force implementation with restored projection data.
	if (m_contactaudit && tn>0.01 && m_mu*m_epsf>0)
	{
		double now=GetFEModel()->GetTime().currentTime;
		auto& point=ss.m_data[m]; mat2d metricInv=point.m_M.inverse();
		vec2d increment=point.m_rs-point.m_rsp;
		double q0=point.m_Lt[0]+m_epsf*(point.m_M[0][0]*increment[0]+point.m_M[0][1]*increment[1]);
		double q1=point.m_Lt[1]+m_epsf*(point.m_M[1][0]*increment[0]+point.m_M[1][1]*increment[1]);
		double qnorm=sqrt(MAX(0.0,q0*(metricInv[0][0]*q0+metricInv[0][1]*q1)+q1*(metricInv[1][0]*q0+metricInv[1][1]*q1)));
		int branch=qnorm>m_mu*tn ? 1 : 0;
		static double checked[2]={-1,-1};
		if (now!=checked[branch] && fabs(qnorm-m_mu*tn)>1e-6*eps)
		{
			checked[branch]=now;
			auto& d=ss.m_data[m]; vec2d saved=d.m_rs; double savedGap=d.m_gap;
			vector<double> fp(ndof),fm(ndof); double err=0,den=0, h=1e-8;
			for (int j=0;j<2;++j)
			{
				d.m_rs=vec2d(saved[0]+h*D1[j],saved[1]+h*D2[j]); d.m_gap=savedGap-h*N[j];
				ContactNodalForce(m,ss,mel,fp);
				d.m_rs=vec2d(saved[0]-h*D1[j],saved[1]-h*D2[j]); d.m_gap=savedGap+h*N[j];
				ContactNodalForce(m,ss,mel,fm);
				for (int i=0;i<2;++i) { double fd=-(fp[i]-fm[i])/(2*h); err+=(fd-ke[i][j])*(fd-ke[i][j]); den+=fd*fd; }
			}
			d.m_rs=saved; d.m_gap=savedGap;
			feLog("CONTACT_TANGENT t=%.12g node=%d relative=%.12g corrected=%d kt=%.12g branch=%d\n",now,ss.NodeIndex(m)+1,sqrt(err/MAX(den,1e-30)),int(m_consistent_tangent),m_epsf,branch);
		}
	}
'''


AUDIT = r'''
	// softmodeling: inspect actual segment/history, not a nearest-facet guess.
	double audit_projection=0;
	if (m_contactaudit)
	{
		double dnmax=0, dtmax=0, slipmax=0;
		int npass = m_btwo_pass ? 2 : 1;
		for (int pass=0; pass<npass; ++pass)
		{
			FESlidingSurface& surf = pass == 0 ? m_ss : m_ms;
			for (int i=0; i<surf.Nodes(); ++i)
			{
				auto& d=surf.m_data[i];
				if (!d.m_pme) continue;
				double pn=MBRACKET(d.m_Lm + scale*d.m_eps*d.m_gap);
				mat2d inv=d.m_M.inverse();
				vec2d dr=d.m_rs-d.m_rsp;
				double ds[2]={d.m_M[0][0]*dr[0]+d.m_M[0][1]*dr[1],d.m_M[1][0]*dr[0]+d.m_M[1][1]*dr[1]};
				double tr[2]={d.m_Lt[0]+m_epsf*ds[0],d.m_Lt[1]+m_epsf*ds[1]};
				auto norm=[&](const double* v) { return sqrt(MAX(0.0,v[0]*(inv[0][0]*v[0]+inv[0][1]*v[1])+v[1]*(inv[1][0]*v[0]+inv[1][1]*v[1]))); };
				double trial=norm(tr), capacity=m_mu*pn;
				if (trial>capacity) { tr[0]*=capacity/trial; tr[1]*=capacity/trial; }
				// Necessary fixed-point test on ACTUAL force traction; rho=250
				// matches the project's existing dimensional projection scale.
				double projected[2]={tr[0]+250.0*ds[0],tr[1]+250.0*ds[1]};
				double radius=norm(projected);
				if (radius>capacity) { projected[0]*=capacity/radius; projected[1]*=capacity/radius; }
				double change[2]={projected[0]-tr[0],projected[1]-tr[1]};
				double dn=fabs(MBRACKET(pn+250.0*d.m_gap)-pn), dt=norm(change), slip=norm(ds);
				dnmax=MAX(dnmax,dn); dtmax=MAX(dtmax,dt); slipmax=MAX(slipmax,(pn>1e-6 ? slip : 0));
				if (pn>1e-6 && tp.currentTime>=0.11249 && tp.currentTime<=0.13751)
					feLog("CONTACT_NODE t=%.12g aug=%d node=%d facet=%d p=%.12g shear=%.12g gap=%.12g slip=%.12g dn=%.12g dt=%.12g kn=%.12g kt=%.12g\n",tp.currentTime,naug+1,surf.NodeIndex(i)+1,d.m_pme->GetID(),pn,norm(tr),d.m_gap,slip,dn,dt,scale*d.m_eps,m_epsf);
			}
		}
		audit_projection=MAX(dnmax,dtmax);
		feLog("CONTACT_AUDIT t=%.12g aug=%d dn=%.12g dt=%.12g slip=%.12g\n",tp.currentTime,naug+1,dnmax,dtmax,slipmax);
	}
'''


def apply(source):
    source=Path(source);cpp=source/'FEBioMech/FESlidingInterface.cpp';header=source/'FEBioMech/FESlidingInterface.h'
    text=cpp.read_text();original=hashlib.sha256(cpp.read_bytes()).hexdigest()
    if 'softmodeling:' in text:raise ValueError('Patch requires pristine upstream source')
    def replace(old,new):
        nonlocal text
        if text.count(old)!=1:raise ValueError(f'Unexpected upstream anchor: {old[:70]}')
        text=text.replace(old,new)
    replace('\tADD_PARAMETER(m_atol', '\tADD_PARAMETER(m_contactaudit, "contact_audit");\n\tADD_PARAMETER(m_consistent_tangent, "consistent_tangent");\n\tADD_PARAMETER(m_strict_augmentation, "strict_augmentation");\n\tADD_PARAMETER(m_atol')
    replace('\tm_mu = 0;', '\tm_contactaudit=false; m_consistent_tangent=false; m_strict_augmentation=false;\n\tm_mu = 0;')
    replace('#include <FECore/FEAnalysis.h>','#include <FECore/FEAnalysis.h>\n#include "FECore/FEException.h"')
    replace('\t// check convergence\n\tbconv = true;',AUDIT+'\n\t// check convergence\n\tbconv = true;')
    replace('\tif (m_naugmax <= naug) bconv = true;', '\tif (m_strict_augmentation && audit_projection>0.001) bconv=false;\n\tif (m_naugmax <= naug) { if (m_strict_augmentation && !bconv) throw IterationFailure(); bconv = true; }')
    for k in range(2):
        old=f'\t\tTt[{k}] = Lt[{k}] + Mk[{k}][0]*(r - rp) + Mk[{k}][1]*(s - sp);'
        replace(old,f'\t\tTt[{k}] = Lt[{k}] + (m_consistent_tangent ? m_epsf : 1.0)*(Mk[{k}][0]*(r - rp) + Mk[{k}][1]*(s - sp));')
    replace('if ((m_mu*m_epsf > 0) && (gap >=0))','if ((m_mu*m_epsf > 0) && (m_consistent_tangent ? tn > 0 : gap >= 0))')
    old='\t\t\t\t\tke[i][j] += m_ktmult*m_epsf*m_mu*tn/sqrt(TMT)*kij;'
    replace(old,old+'\n\t\t\t\t\t// softmodeling: derivative of Coulomb capacity mu*tn.\n\t\t\t\t\tif (m_consistent_tangent) ke[i][j] -= eps*(Tt[0]*D1[i]+Tt[1]*D2[i])*N[j];')
    anchor='}\n\n//-----------------------------------------------------------------------------\n\nbool FESlidingInterface::Augment'
    replace(anchor,TANGENT_CHECK+'\n'+anchor)
    h=header.read_text();anchor='\tdouble\t\t\tm_atol;'
    if h.count(anchor)!=1:raise ValueError('Unexpected header')
    header.write_text(h.replace(anchor,'\tbool m_contactaudit, m_consistent_tangent, m_strict_augmentation;\n'+anchor))
    cpp.write_text(text)
    return {'upstream_cpp_sha256':original,'patched_cpp_sha256':hashlib.sha256(cpp.read_bytes()).hexdigest(),
            'patched_header_sha256':hashlib.sha256(header.read_bytes()).hexdigest(),
            'recipe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
