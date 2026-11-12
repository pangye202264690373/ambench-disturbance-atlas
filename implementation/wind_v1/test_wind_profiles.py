import sys,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
import numpy as np,torch
from wind_profile import WindProfile,WindSequence,derive_wind_seed,KINDS
from wind_runtime import WindWrenchAdapter,world_to_link,link_to_world
from types import SimpleNamespace
from dataclasses import dataclass
T=np.arange(1200)/120;seed=derive_wind_seed('diagnostic',53060000)
state=np.random.get_state();checks=[]
for kind in sorted(KINDS):
 fcap=0 if kind in ['off','random_torque'] else 5.;tcap=.2 if kind in ['random_torque','combined'] else 0
 p=WindProfile(kind,fcap,tcap,startup_s=0 if kind=='constant' else .5,ramp_s=0 if kind=='constant' else .25);a=WindSequence(p,seed,10);b=WindSequence(p,seed,10);F,Q=a.sample(T);F2,Q2=b.sample(T)
 assert np.array_equal(F,F2) and np.array_equal(Q,Q2)
 assert np.isfinite(F).all() and np.isfinite(Q).all()
 assert np.linalg.norm(F,axis=1).max()<=fcap+1e-12
 assert np.linalg.norm(Q,axis=1).max()<=tcap+1e-12
 for i in [0,2,120,881]:
  fs,qs=a.sample(T[i]);assert np.array_equal(fs,F[i]) and np.array_equal(qs,Q[i])
 for knot in np.arange(1.2,9.9,1.2):
  l,_=a.sample(knot-1e-7);r,_=a.sample(knot+1e-7);assert np.linalg.norm(l-r)<1e-4
 checks.append({'kind':kind,'max_force_n':float(np.linalg.norm(F,axis=1).max()),'max_torque_nm':float(np.linalg.norm(Q,axis=1).max()),'repeat_sha256':hashlib.sha256(np.c_[F,Q].tobytes()).hexdigest()})
state2=np.random.get_state();assert state[0]==state2[0] and np.array_equal(state[1],state2[1]) and state[2:]==state2[2:]
a=WindSequence(WindProfile('combined',2,.2),seed,10);b=WindSequence(WindProfile('combined',10,.2),seed,10);fa,ta=a.sample(T);fb,tb=b.sample(T);np.testing.assert_allclose(fb,5*fa,rtol=1e-14,atol=1e-14);assert np.array_equal(ta,tb)
c=WindSequence(WindProfile('combined',2,.2),derive_wind_seed('diagnostic',53060000,1),10);assert not np.array_equal(a.sample(T)[0],c.sample(T)[0])
assert len({derive_wind_seed(x,1) for x in ['train','eval','diagnostic']})==3
q=torch.tensor([[2**-.5,0.,0.,2**-.5]]);v=torch.tensor([[2.,0.,0.]])
np.testing.assert_allclose(world_to_link(q,v).numpy(),[[0.,-2.,0.]],atol=1e-6)
np.testing.assert_allclose(link_to_world(q,world_to_link(q,v)).numpy(),v.numpy(),atol=1e-6)
@dataclass
class Command:
 force_b:torch.Tensor
 torque_b:torch.Tensor
class IO:
 control_body_idx=0
 def apply(self,c):
  composer.composed_force_as_torch[:,0,:].copy_(c.force_b);composer.composed_torque_as_torch[:,0,:].copy_(c.torque_b)
composer=SimpleNamespace(composed_force_as_torch=torch.zeros(1,1,3),composed_torque_as_torch=torch.zeros(1,1,3))
body=torch.zeros(1,1,13);body[:,0,3:7]=q
io=IO();env=SimpleNamespace(cfg=SimpleNamespace(enable_wind_effect=False,sim=SimpleNamespace(dt=1/120)),robot_io=io,num_envs=1,robot=SimpleNamespace(data=SimpleNamespace(body_link_state_w=body),permanent_wrench_composer=composer,body_names=['base_link']),control_pipeline=SimpleNamespace(last_command=None))
base=Command(torch.tensor([[1.,2.,3.]]),torch.tensor([[.1,.2,.3]]));wind=WindWrenchAdapter(env,WindProfile('constant',2,0,startup_s=0,ramp_s=0),seed,10)
io.apply(base);np.testing.assert_allclose(composer.composed_force_as_torch.numpy(),[[[1,0,3]]],atol=1e-6);assert torch.equal(base.force_b,torch.tensor([[1.,2.,3.]]));assert torch.equal(composer.composed_torque_as_torch[:,0,:],base.torque_b)
wind.reset_episode(seed);io.apply(base);assert wind.last_record['physics_step']==[0];wind.detach();assert not hasattr(io,'_wind_wrench_adapter')
wind=WindWrenchAdapter(env,WindProfile('random_torque',0,.2),seed,10)
for _ in range(200):io.apply(base)
assert torch.equal(composer.composed_force_as_torch[:,0,:],base.force_b)
assert not torch.equal(composer.composed_torque_as_torch[:,0,:],base.torque_b)
wind.detach();wind=WindWrenchAdapter(env,WindProfile(),seed,10);io.apply(base);assert torch.equal(composer.composed_force_as_torch[:,0,:],base.force_b);assert torch.equal(composer.composed_torque_as_torch[:,0,:],base.torque_b);wind.detach()
for kwargs in [{'kind':'unknown'},{'force_peak_n':-1},{'knot_interval_s':0},{'kind':'gust','torque_peak_nm':.1}]:
 try:WindProfile(**kwargs);raise AssertionError('Bad config accepted')
 except ValueError:pass
report={'passed':True,'source_hashes':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['wind_profile.py','wind_runtime.py']},'backend':'CPU signals and mocked RobotIO/composer, not GPU gate','checks':checks,'global_rng_unchanged':True,'same_seed_repeat':True,'strength_only_scaling':True,'different_seed_changes_sequence':True,'split_seed_separation':True,'coordinate_transform_90deg':True,'controller_wrench_preserved_and_added':True,'zero_wind_passthrough':True,'reset_rewinds':True,'force_and_torque_readback_checked':True,'training':False}
(P/'cpu_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
