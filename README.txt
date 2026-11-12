AMBench 扰动与随机化视觉看板

内容：12任务原生场景随机化、按钮/孔尺寸端点、固定历史cmd的单因素物理回放。
运行：python3 -m http.server 8786，然后打开 http://localhost:8786/
数据入口：data/catalog.json、data/configuration.json、data/provenance.json、data/validation.json。

证据口径
- 场景截图为真实Isaac RGB；无AI生成或示意图替代。
- PressButton回放使用E11 seed53060000，TossBall使用E17 seed53100000，WipeWindow使用E11 seed53110000。
- requested_action在120Hz物理步前输入；测量同一步之后的实际末端位置。
- baseline保留重力、接触、机器人-机械臂耦合，只关闭额外噪声/旋翼限制与动态/气动/恒力。
- dynamics条件包含推力饱和，分离动态增量时应与saturation条件比较。
- ground/wall/drag从当前气动实现分别置零其他项得到，属于诊断开关，未修改原正式测评代码。
- 采用固定短窗口，并抑制任务成功后的自动reset；不是完整评测回合，不报告新成功率。
- 未加载任何神经策略或训练权重。
- 旋翼速率限制是归一化转速每秒变化率，不是机体加速度。
- 可视化曲线降采样；P95与峰值统计来自完整120Hz记录。

本看板仅描述本项目核验的部署版本，不代表所有上游AMBench或Isaac Lab版本。
仅发布可视化资源、参数和汇总，不包含集群账号、凭据、私有路径或训练权重。
