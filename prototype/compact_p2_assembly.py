"""Opt-in equivalent local-P2 assembly without padded P1 eigensystems/COO.

Only the current local viscoelastic P2 layout is supported. The isolated
evaluate override is deliberately bounded to that layout until path qualification;
it never replaces pressure-block or general IPC assembly.
"""
import numpy as np
from scipy.sparse import coo_matrix


def compact_matrices(k,dofs,total,linear,active,*,projected=False,with_metric=False):
    groups=((linear,6),(active,12))
    values=[];projected_values=[];rows=[];cols=[]
    for ids,width in groups:
        local=k[ids,:width,:width];dd=dofs[ids,:width]
        rows.append(np.broadcast_to(dd[:,:,None],local.shape).ravel())
        cols.append(np.broadcast_to(dd[:,None,:],local.shape).ravel())
        if not projected:values.append(local.ravel())
        if projected or with_metric:
            w,v=np.linalg.eigh(local)
            projected_values.append(((v*np.maximum(w,0)[:,None,:])@v.transpose(0,2,1)).ravel())
    rc=(np.concatenate(rows),np.concatenate(cols));shape=(total,total)
    def assemble(v):return coo_matrix((np.concatenate(v),rc),shape=shape).tocsc()
    psd=assemble(projected_values) if projected or with_metric else None
    exact=psd if projected else assemble(values)
    return exact,psd


class CompactP2AssemblyMixin:
    def evaluate(self,x,lag=None,tangent=True,projected=True,*,with_metric=False,with_pressure_blocks=False):
        if self.element!='p2-local-visco' or self.nodal_volume is not None or with_pressure_blocks:
            raise ValueError('Compact assembly requires local viscoelastic P2 without pressure blocks')
        if with_metric and not tangent:raise ValueError('Metric requires tangent')
        u=(x[:self.n]-self.mesh.reference_xy).ravel()
        r,k,dofs,energy,minj=self.enrichment.bulk(u,with_tangent=tangent)
        g=np.zeros(self.total);np.add.at(g,dofs.ravel(),r.ravel())
        c=self.collisions(x);sx=x[self.surface]
        energy+=self.barrier(c,self.cmesh,sx)
        cg=self.barrier.gradient(c,self.cmesh,sx)
        if lag is not None:
            delta=(x-self.previous)[self.surface]
            energy+=self.friction(lag,self.cmesh,delta)
            cg+=self.friction.gradient(lag,self.cmesh,delta)
        g[self.surface_dofs]+=cg
        h=metric=None
        if tangent:
            h,metric=compact_matrices(k,dofs,self.total,self.linear_elements,self.active_elements,
                                      projected=projected,with_metric=with_metric)
            def contact(clamp):
                projection=self.ipc.PSDProjectionMethod.CLAMP if clamp else self.ipc.PSDProjectionMethod.NONE
                ch=self.barrier.hessian(c,self.cmesh,sx,projection)
                if lag is not None:ch+=self.friction.hessian(lag,self.cmesh,(x-self.previous)[self.surface],projection)
                ch=ch.tocoo()
                return coo_matrix((ch.data,(self.surface_dofs[ch.row],self.surface_dofs[ch.col])),
                                   shape=(self.total,self.total)).tocsc()
            h+=contact(projected)
            if with_metric:metric=h if projected else metric+contact(True)
        answer=(energy,g,h,minj)
        return answer+(metric,) if with_metric else answer
