Wind wrench v1 / 风扰动实现

范围：机身base_link的等效世界系外力F(t)与外力矩tau(t)，单位N、N m。
不是风速模型，不对球或其他物体额外施加风阻。现有full默认配置和旧实验不改动。

信号：off、constant、gust、random_direction、random_strength、random_torque、combined。
固定方向可配置fixed_direction_w并归一化；random_direction/combined当前在世界XY水平面变化。
方向/强度/力矩以1.2秒随机节点做半余弦平滑插值。阵风为平滑sin²脉冲，持续0.6～1.2秒、间隔0.4～1秒，首次约0.7～1秒。
复现身份包含profile、seed、代码版本和预登记duration_s；未来采集/评测应使用固定任务时限初始化，不得用回合成功终止的长度重建风序列。
时变模式开始0.5秒静置并用0.25秒平滑起振；constant_2n从t=0生效，用于旧恒力路径回归。
combined力幅值=Fcap*(0.25+0.75*pulse)*random_strength，力矩用独立随机流。
峰值上限相同不等于RMS或冲量相同；不同强度复用同一归一化波形，力矩保持同一0.2N m上限波形。

接入：
  cfg.enable_wind_effect=False  # 关闭旧恒力分支，避免重复叠加
  env.reset(seed=scene_seed)
  wind=WindWrenchAdapter(env, profile, wind_seed, duration_s, audit=True)
  env.step(action)             # 原控制器和旋翼分配完成后，自动叠加外部wrench
每次重置环境后必须调用wind.reset_episode(new_wind_seed)，或detach后重新创建adapter。
结束使用wind.detach()。不改全局类，不改模型输入，也不把风seed/真值加入策略观测。

审计：adapter读取真实permanent_wrench_composer的force/torque，减去控制器命令并变回世界系。
记录请求与回读的F/torque、叠加前后总wrench、物理步、seed和源哈希。不能把仅计划的曲线冒充实际施加。
F/torque时间是区间起点，实际位姿是该步之后；不使用墙钟或渲染耗时驱动风。

复现：诊断/未来train/eval采用不同seed命名域，不消耗环境或策略的全局随机数。
正式采集和正式评测尚未启动；seed_registry.json中的未来eval名单是未执行的预登记方案。
CPU tests: python test_wind_profiles.py（NumPy和Torch CPU环境即可，无需GPU）。
当前回放使用历史cmd，不运行神经策略；结果不能称为模型成功率或真实风速鲁棒性。

本次cmd是120Hz的末端目标位姿与夹爪指令，低层控制仍反馈运行，不是冻结电机推力。仅诊断worker抑制成功后的自动reset以保持相同窗口；可复用模块本身不修改成功判据。
