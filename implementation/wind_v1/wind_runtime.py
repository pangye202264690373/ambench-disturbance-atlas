"""Per-environment opt-in adapter on the existing RobotIO.apply interface.
The controller/rotor allocation completes first; external wrench is then added.
Call reset_episode after every environment reset. No upstream source is mutated.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import torch
from wind_profile import WindProfile,WindSequence

def world_to_link(q,v):
    # q is wxyz, maps local to world. Rotation only, same origin for torque.
    qw=q[:,:1];qv=q[:,1:];cross=torch.cross(qv,v,dim=-1)
    return v+2*(-qw*cross+torch.cross(qv,cross,dim=-1))

def link_to_world(q,v):
    inv=q.clone();inv[:,1:]*=-1
    return world_to_link(inv,v)

class WindWrenchAdapter:
    def __init__(self,env,profile:WindProfile,seeds,duration_s:float,audit=True):
        if env.cfg.enable_wind_effect:raise ValueError('Disable native constant-wind branch to avoid double counting')
        if hasattr(env.robot_io,'_wind_wrench_adapter'):raise RuntimeError('A wind adapter is already attached')
        self.env=env;self.io=env.robot_io;self.profile=profile;self.duration_s=duration_s;self.dt=float(env.cfg.sim.dt)
        self.audit=audit;self.original_apply=self.io.apply;self.sequences=[];self.steps=np.zeros(env.num_envs,dtype=np.int64);self.last_record=None
        self.reset_episode(seeds)
        self.io._wind_wrench_adapter=self;self.io.apply=self.apply
    def reset_episode(self,seeds):
        seeds=[int(seeds)] if isinstance(seeds,(int,np.integer)) else [int(s) for s in seeds]
        if len(seeds)!=self.env.num_envs:raise ValueError('One independent wind seed per environment is required')
        self.sequences=[WindSequence(self.profile,s,self.duration_s) for s in seeds];self.steps[:]=0;self.last_record=None
    def apply(self,command):
        body=int(self.io.control_body_idx);q=self.env.robot.data.body_link_state_w[:,body,3:7]
        if not torch.isfinite(q).all() or float((q.norm(dim=-1)-1).abs().max())>1e-3:raise RuntimeError('Invalid target body quaternion')
        q=q/q.norm(dim=-1,keepdim=True)
        requested=[seq.sample(int(step)*self.dt) for seq,step in zip(self.sequences,self.steps)]
        fw=torch.as_tensor(np.stack([x[0] for x in requested]),dtype=command.force_b.dtype,device=command.force_b.device)
        tw=torch.as_tensor(np.stack([x[1] for x in requested]),dtype=command.torque_b.dtype,device=command.torque_b.device)
        fb=world_to_link(q,fw);tb=world_to_link(q,tw)
        base_f=command.force_b.clone();base_t=command.torque_b.clone()
        applied=replace(command,force_b=base_f+fb,torque_b=base_t+tb)
        if not torch.isfinite(applied.force_b).all() or not torch.isfinite(applied.torque_b).all():raise RuntimeError('Nonfinite composed wrench')
        # Preserve observer access to the ACTUAL command without changing controller outputs.
        self.env.control_pipeline.last_command=applied
        self.original_apply(applied)
        composer=self.env.robot.permanent_wrench_composer
        read_f=composer.composed_force_as_torch[:,body,:].clone();read_t=composer.composed_torque_as_torch[:,body,:].clone()
        error_f=float((read_f-applied.force_b).abs().max());error_t=float((read_t-applied.torque_b).abs().max())
        if self.audit and (error_f>1e-4 or error_t>1e-5):raise RuntimeError(('Simulator wrench readback mismatch',error_f,error_t))
        # Difference to controller output recovers what this layer actually added.
        actual_fw=link_to_world(q,read_f-base_f);actual_tw=link_to_world(q,read_t-base_t)
        plain=lambda v:v.detach().cpu().tolist()
        self.last_record={'interval_start_s':(self.steps*self.dt).tolist(),'physics_step':self.steps.tolist(),'wind_seed':[s.seed for s in self.sequences],'target_body':self.env.robot.body_names[body],'requested_force_w':plain(fw),'requested_torque_w':plain(tw),'added_force_b':plain(fb),'added_torque_b':plain(tb),'readback_added_force_w':plain(actual_fw),'readback_added_torque_w':plain(actual_tw),'controller_force_b':plain(base_f),'controller_torque_b':plain(base_t),'composed_force_b':plain(read_f),'composed_torque_b':plain(read_t),'readback_max_error_force_n':error_f,'readback_max_error_torque_nm':error_t}
        self.steps+=1
    def detach(self):
        if getattr(self.io,'_wind_wrench_adapter',None) is self:
            self.io.apply=self.original_apply;del self.io._wind_wrench_adapter
