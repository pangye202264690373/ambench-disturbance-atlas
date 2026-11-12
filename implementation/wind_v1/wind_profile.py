"""Versioned, seeded equivalent wind-wrench signals (N, N m, world frame).
No global RNG changes, no wall-clock time, no aerodynamic wind-speed claim.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib,json,math
import numpy as np

VERSION='wind-wrench-v1'
KINDS={'off','constant','gust','random_direction','random_strength','random_torque','combined'}

@dataclass(frozen=True)
class WindProfile:
    kind: str='off'
    force_peak_n: float=0.0
    torque_peak_nm: float=0.0
    knot_interval_s: float=1.2
    startup_s: float=0.5
    ramp_s: float=0.25
    fixed_direction_w: tuple[float,float,float]=(1.0,0.0,0.0)
    def __post_init__(self):
        direction=np.asarray(self.fixed_direction_w,dtype=float)
        if direction.shape!=(3,) or not np.isfinite(direction).all() or np.linalg.norm(direction)<=0:raise ValueError('fixed_direction_w must be a finite nonzero 3-vector')
        if self.kind not in KINDS:raise ValueError('Unknown wind kind')
        for key in ['force_peak_n','torque_peak_nm','startup_s','ramp_s']:
            v=getattr(self,key)
            if not math.isfinite(v) or v<0:raise ValueError(key+' must be finite and nonnegative')
        if not math.isfinite(self.knot_interval_s) or self.knot_interval_s<=0:raise ValueError('knot_interval_s must be positive')
        if self.kind not in {'random_torque','combined'} and self.torque_peak_nm!=0:raise ValueError('torque must be zero for this kind')
        if self.kind in {'off','random_torque'} and self.force_peak_n!=0:raise ValueError('force must be zero for this kind')
        if self.kind=='off' and self.torque_peak_nm!=0:raise ValueError('off must have zero torque')
    def to_dict(self):return {'version':VERSION,**asdict(self)}

def derive_wind_seed(split: str, scene_seed: int, replicate: int=0) -> int:
    if split not in {'diagnostic','train','eval'}:raise ValueError('Unknown split')
    if any(isinstance(x,bool) or not isinstance(x,(int,np.integer)) or x<0 for x in [scene_seed,replicate]):raise ValueError('Seeds and replicate must be nonnegative integers')
    text=f'{VERSION}|{split}|{int(scene_seed)}|{int(replicate)}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'little')

class WindSequence:
    def __init__(self, profile: WindProfile, seed: int, duration_s: float):
        if isinstance(seed,bool) or not isinstance(seed,(int,np.integer)) or not 0<=int(seed)<2**64:raise ValueError('seed outside uint64')
        if not math.isfinite(duration_s) or duration_s<=0:raise ValueError('duration must be positive')
        self.profile=profile;self.seed=int(seed);self.duration_s=float(duration_s)
        # Independent streams retain the same shape when only physical strength changes.
        rngs=[np.random.Generator(np.random.PCG64(s)) for s in np.random.SeedSequence(self.seed).spawn(4)]
        n=int(math.ceil(duration_s/profile.knot_interval_s))+3
        self.heading=rngs[0].uniform(-math.pi,math.pi,n);self.heading[0]=0.
        self.amplitude=rngs[1].uniform(.25,1.,n)
        torque=rngs[2].normal(size=(n,3));torque/=np.linalg.norm(torque,axis=1,keepdims=True)
        self.torque_knots=torque*rngs[2].uniform(.2,1.,(n,1))
        events=[];start=.7+float(rngs[3].uniform(0,.3))
        while start<duration_s+1.5:
            width=float(rngs[3].uniform(.6,1.2));events.append((start,width));start+=width+float(rngs[3].uniform(.4,1.0))
        self.events=events
    @staticmethod
    def smooth(u):return .5-.5*np.cos(np.pi*np.clip(u,0.,1.))
    def sample(self,times):
        t=np.asarray(times,dtype=np.float64)
        if np.any(~np.isfinite(t)) or np.any(t<0) or np.any(t>self.duration_s+1e-9):raise ValueError('Time outside registered episode')
        scalar=t.ndim==0;t=t.reshape(-1);p=self.profile
        k=np.floor(t/p.knot_interval_s).astype(int);u=self.smooth(t/p.knot_interval_s-k)
        delta=np.arctan2(np.sin(self.heading[k+1]-self.heading[k]),np.cos(self.heading[k+1]-self.heading[k]))
        angle=self.heading[k]+u*delta
        amplitude=(1-u)*self.amplitude[k]+u*self.amplitude[k+1]
        torque=(1-u[:,None])*self.torque_knots[k]+u[:,None]*self.torque_knots[k+1]
        pulse=np.zeros_like(t)
        for start,width in self.events:
            z=(t-start)/width;mask=(z>=0)&(z<=1);pulse[mask]=np.maximum(pulse[mask],np.sin(np.pi*z[mask])**2)
        gate=np.where(t>=p.startup_s,1.,0.) if p.ramp_s==0 else self.smooth((t-p.startup_s)/p.ramp_s)
        f=np.zeros((len(t),3));tau=np.zeros_like(f);direction=np.asarray(p.fixed_direction_w,dtype=float);direction=direction/np.linalg.norm(direction)
        if p.kind=='constant':f=p.force_peak_n*gate[:,None]*direction
        elif p.kind=='gust':f=(p.force_peak_n*pulse*gate)[:,None]*direction
        elif p.kind=='random_direction':f[:,:2]=p.force_peak_n*gate[:,None]*np.stack([np.cos(angle),np.sin(angle)],axis=-1)
        elif p.kind=='random_strength':f=(p.force_peak_n*amplitude*gate)[:,None]*direction
        elif p.kind=='random_torque':tau=p.torque_peak_nm*torque*gate[:,None]
        elif p.kind=='combined':
            magnitude=p.force_peak_n*(.25+.75*pulse)*amplitude*gate
            f[:,:2]=magnitude[:,None]*np.stack([np.cos(angle),np.sin(angle)],axis=-1)
            tau=p.torque_peak_nm*torque*gate[:,None]
        return (f[0],tau[0]) if scalar else (f,tau)
    def manifest(self):
        return {'profile':self.profile.to_dict(),'wind_seed':self.seed,'duration_s':self.duration_s,'coordinate_frame':'world','force_units':'N','torque_units':'N m','torque_reference':'target body link origin, same reference as existing RobotIO wrench','numpy_version':np.__version__,'signal_model':'prescribed equivalent wrench; not calibrated airspeed/CFD','events_s':self.events}
