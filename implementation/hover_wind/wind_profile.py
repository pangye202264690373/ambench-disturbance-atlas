"""Independent wind factors for hover diagnostics. Units N/Nm, world coordinates."""
from dataclasses import dataclass,asdict
import numpy as np
import math
VERSION='hover-independent-v1'
@dataclass
class WindProfile:
 force_peak_n:float
 torque_peak_nm:float
 startup_s:float
 ramp_s:float
 mean_force:dict
 gust:dict
 random_direction:dict
 random_strength:dict
 constant_torque:dict
 random_torque:dict
 def __post_init__(self):
  for v in [self.force_peak_n,self.torque_peak_nm,self.startup_s,self.ramp_s]:assert math.isfinite(v) and v>=0
  for d in [self.random_direction,self.random_strength,self.random_torque]:assert d['interval_s']>0
  for key in ['duration_s','gap_s']:
   a,b=self.gust[key];assert 0<a<=b
  a,b=self.random_strength['multiplier_range'];assert 0<=a<=b
  assert 0<=self.gust['peak_n'] and self.gust['first_start_s']>=0
  assert 0<=self.random_torque['peak_nm']
  assert np.linalg.norm(self.random_direction['fixed_direction_w'])>0
 def to_dict(self):return {'version':VERSION,**asdict(self)}
class WindSequence:
 def __init__(self,profile,seed,duration_s):
  self.profile=profile;self.seed=int(seed);self.duration_s=float(duration_s);assert self.duration_s>0
  rngs=[np.random.default_rng(s) for s in np.random.SeedSequence(self.seed).spawn(4)];self.knots={}
  for name,rng in zip(['random_direction','random_strength','random_torque'],rngs):
   d=getattr(profile,name);n=int(np.ceil(duration_s/d['interval_s']))+3
   if name=='random_direction':v=np.deg2rad(rng.uniform(*d['yaw_range_deg'],size=n))
   elif name=='random_strength':v=rng.uniform(*d['multiplier_range'],size=n)
   else:
    v=rng.normal(size=(n,3))*np.asarray(d['axis_weights']);v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12);v*=rng.uniform(.2,1,size=(n,1))*d['peak_nm']
   self.knots[name]=v
  self.events=[];t=profile.gust['first_start_s']
  while t<duration_s:
   width=rngs[3].uniform(*profile.gust['duration_s']);self.events.append((t,width));t+=width+rngs[3].uniform(*profile.gust['gap_s'])
 @staticmethod
 def smooth(x):return .5-.5*np.cos(np.pi*np.clip(x,0,1))
 def interpolate(self,name,t):
  interval=getattr(self.profile,name)['interval_s'];k=np.floor(t/interval).astype(int);u=self.smooth(t/interval-k);v=self.knots[name];delta=v[k+1]-v[k]
  if name=='random_direction':delta=np.arctan2(np.sin(delta),np.cos(delta))
  return v[k]+(u[:,None] if v.ndim==2 else u)*delta
 def sample(self,times):
  t=np.asarray(times,dtype=float);scalar=t.ndim==0;t=t.reshape(-1);assert np.isfinite(t).all() and (t>=0).all() and (t<=self.duration_s+1e-9).all();p=self.profile
  gate=self.smooth((t-p.startup_s)/p.ramp_s) if p.ramp_s else (t>=p.startup_s).astype(float)
  f=np.zeros((len(t),3));tau=np.zeros_like(f)
  if p.mean_force['enabled']:
   d=np.asarray(p.mean_force['direction_w'],float);assert np.linalg.norm(d)>0;f+=p.mean_force['magnitude_n']*d/np.linalg.norm(d)
  d=p.random_direction
  if d['enabled']:
   angle=self.interpolate('random_direction',t);direction=np.c_[np.cos(angle),np.sin(angle),np.zeros(len(t))]
  else:
   v=np.asarray(d['fixed_direction_w'],float);direction=np.tile(v/np.linalg.norm(v),(len(t),1))
  pulse=np.zeros(len(t))
  if p.gust['enabled']:
   for start,width in self.events:
    z=(t-start)/width;mask=(z>=0)&(z<=1);pulse[mask]=np.maximum(pulse[mask],np.sin(np.pi*z[mask])**2)
  magnitude=p.gust['peak_n']*pulse
  if p.random_strength['enabled']:magnitude=(magnitude+p.random_strength['background_n'])*self.interpolate('random_strength',t)
  f+=magnitude[:,None]*direction
  if p.constant_torque['enabled']:tau+=np.asarray(p.constant_torque['vector_w_nm'],float)
  if p.random_torque['enabled']:tau+=self.interpolate('random_torque',t)
  for arr,cap in [(f,p.force_peak_n),(tau,p.torque_peak_nm)]:arr*=np.minimum(1,cap/np.maximum(np.linalg.norm(arr,axis=1),1e-12))[:,None];arr*=gate[:,None]
  return (f[0],tau[0]) if scalar else (f,tau)
 def manifest(self):return {'profile':self.profile.to_dict(),'wind_seed':str(self.seed),'duration_s':self.duration_s,'events_s':self.events,'composition':'mean force + directed (gust + background)*random strength; independent torque; final norm caps','units':['N','N m'],'frame':'world','numpy_version':np.__version__}
