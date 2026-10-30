
### refiner 反推实验（04:10-04:40，负结果）
- 成功重建 notoverkil refiner 架构（ResNet34(3,4,6,3) backbone + U-Net decoder，输入 2 通道 256，24.44M 参数，strict load OK；decoder BN 为 track_running_stats=False）。
- 但应用校准失败：9 组预处理组合下 refined mask 与 ref 掩膜 IoU 仅 0.10-0.33（baseline 0.89）→ 裁剪/归一化约定未知，无法在时限内复现其 refiner 行为。
- 结论：放弃 refiner 精修，vote15 候选直接使用 consensus 掩膜（与 ref 集合一致、边界略粗）。
