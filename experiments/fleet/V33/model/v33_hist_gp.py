"""Small additive Gaussian process for ordered sensor histograms, research only."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

START = (23,37,54,80,129,154)

def extract(data):
    groups=[]
    for start in START:
        width=9 if start==54 else 10
        x=data[[f"s{i:03d}" for i in range(start,start+width)]].to_numpy(float)
        total=x.sum(axis=1)
        ok=np.isfinite(x).all(1)&(x>=0).all(1)&(total>0)
        frac=x/np.where(ok,total,np.nan)[:,None]
        # Shape and within-histogram location both vary continuously.
        groups.append(np.column_stack([np.sqrt(frac),frac.cumsum(1)[:,:-1]]))
        if start==80: exposure=np.log1p(np.where(ok,total,np.nan))[:,None]
    return groups+[exposure]

def select(data,rows):
    return [x[rows] for x in data]

class GaussianSeverity:
    def blocks(self, features):
        result=[]
        for x,base,scale in zip(features,self.training,self.scale):
            distance=cdist(x,base,'sqeuclidean')
            result.append(np.nan_to_num(np.exp(-distance/(2*scale)),nan=0.))
        return np.asarray(result)

    def fit(self,features,y):
        self.training=features
        self.scale=[]
        ind=np.triu_indices(len(y),1)
        for x in features:
            ds=cdist(x,x,'sqeuclidean')[ind]
            ds=ds[np.isfinite(ds)]
            md=np.median(ds) if len(ds) else 1.
            self.scale.append(max(md,1.e-6))
        blocks=self.blocks(features)
        self.center=np.mean(y)
        z=y-self.center
        unit=np.eye(len(y))
        def value(logtheta,final=False):
            theta=np.exp(logtheta)
            cov=np.einsum('i,ijk->jk',theta[:-1],blocks)+theta[-1]*unit
            factor=cho_factor(cov,lower=True,check_finite=False)
            alpha=cho_solve(factor,z,check_finite=False)
            if final: return alpha
            loss=.5*z@alpha+np.log(np.diag(factor[0])).sum()+.5*(logtheta@logtheta)/4
            inv=cho_solve(factor,unit,check_finite=False)-np.outer(alpha,alpha)
            grad=.5*theta*np.r_[np.einsum('ij,kij->k',inv,blocks),np.trace(inv)]+logtheta/4
            return loss,grad
        res=minimize(value,np.zeros(len(blocks)+1),jac=True,method='L-BFGS-B',
                     bounds=[(-5.,5.)]*len(blocks)+[(-4.,3.)],options={'maxiter':35})
        self.amplitude=np.exp(res.x[:-1])
        self.alpha=value(res.x,True)
        self.bounds=(min(y),max(y))
        return self

    def predict(self,features):
        out=np.full(len(features[0]),np.nan)
        for start in range(0,len(out),800):
            rows=slice(start,start+800)
            vals=select(features,rows)
            valid=sum(np.isfinite(x).all(1).astype(int) for x in vals[:-1])>=3
            blocks=self.blocks(vals)
            pred=np.einsum('i,ijk->jk',self.amplitude,blocks)@self.alpha+self.center
            out[rows]=np.where(valid,np.clip(pred,*self.bounds),np.nan)
        return out
