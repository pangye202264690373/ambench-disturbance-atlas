固定目标悬停 / 独立风因素诊断

parameters.json：每个风因素独立 enabled 开关及参数。
mean_force：恒定风 magnitude_n、direction_w。
gust：peak_n、duration_s、gap_s、first_start_s。
random_direction：interval_s、yaw_range_deg、fixed_direction_w（关闭随机方向时使用）。
random_strength：interval_s、multiplier_range、background_n。
constant_torque：世界系 vector_w_nm。
random_torque：interval_s、peak_nm、axis_weights。
force_peak_n / torque_peak_nm：所有分量叠加后的范数硬上限。
startup_s / ramp_s：全局静置和起振。
风力 = 恒定风 +（阵风 + 随机强度背景）×强度倍率×时变方向；力矩单独叠加。
关闭随机强度时只有阵风分量，不额外添加背景。恒定风方向独立于随机方向。
所有单位N/Nm，世界系，非m/s、非CFD。力矩箭头表示转轴，正负遵守右手定则。

修改参数后，在新的运行目录运行 python configure.py --parameters parameters.json。
随后重新做CPU校验与提交预算检查。已经提交的目录禁止修改，以保留可复现性。
风序列由固定wind_seed和duration生成，与墙钟或任务成功时刻无关。

此场景固定初始末端绝对位姿目标，经原IK/PID控制器维持悬停；机器人刚体不锁死。
真实漂移可见。没有神经策略推理、没有任务成功率评分。
无风与组合风使用同一场景seed、相同固定目标。原full其他物理保持。
wind_worker.py保存真实RGB、120Hz位姿/wrench/rotor记录与hold_target.json。
annotate_video.py OUTPUT目录：将实际回读力/力矩箭头叠加到真实录像旁的世界坐标示意面板。
独立固定比例5px/N、250px/Nm；不是将录像画面虚构成仿真。
