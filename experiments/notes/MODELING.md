# MODELING — 自训 YOLO11-seg + crop refiner（task-4 / task-9）

> 目标：不再依赖公开权重，自训一条 **YOLO11-seg 检测器 + seed-conditioned crop refiner** 的实例分割流水线；
> 在 harness 官方 pooled PQ（多标注者聚合）下给出 fold0 验证数字，并产出可提交/可集成的模型与 submission.csv。
> 背景事实见 FACTS.md（公开 0.70 是旧 Dice 指标，PQ 下同一 pipeline 仅 0.34；人类互评 pooled PQ ≈ 0.34）。

## 1. 数据与折分
- 707 张物理 jpeg（2048²），1154 条 annotator-image 记录（同一物理图 1~3 位标注者），8199 个实例。
- **折分：harness `work/cache/folds.json`（按物理 jpeg 分组, seed=42）**，fold0 = 142 张物理图。
  早期用过本地 fallback（`work/cache/folds_modeling.json`，seed=42 分组），harness 折分就绪后已切换并重建数据集。
- 训练标签来源：**全部 1154 条标注记录**（`--label-source all`）：同一物理图的每条 annotator 记录都作为一个训练样本（文件名 `{stem}__{annotator}`），让模型学习标注者共识几何；fold 内不允许同一物理图跨 train/val。
- 图像预处理：2048 → 1024 预缩放 JPEG（q95）写 `work/yolo_data/images_all_1024/`，各折 hardlink（不额外占盘）。
- GT 掩膜一律用 **pycocotools frPyObjects** 栅格化（与官方 GT 逐像素一致，勿用 cv2.fillPoly）。

## 2. 模型
### 2.1 检测器：YOLO11-seg（ultralytics 8.4.191）
- 单类 filament；多边形标签 = 每实例最大外轮廓多边形，归一化 2048 坐标。
- 训练配置（本地主线）：`yolo11n-seg.pt` 迁移学习，imgsz=1024, batch=8, AMP, mask_ratio=2,
  workers=8, mosaic=0（小目标不用 mosaic）, fliplr/flipud=0.5, scale=0.20, translate=0.05, degrees=0,
  hsv_v=0.25, 20 epochs, cache=disk, val=False（用 PQ 直接选模型）。
- 备选/云端：`yolo11s-seg.pt` 30 epochs（本地 30W 功耗墙解除后实测 s-seg@1024 仍 ~3.5s/it → 3.4h，故本地先跑 n-seg，s-seg 交云端 T4）。

### 2.2 精修器：seed-conditioned ResNet18 U-Net（CropUNet）
- 结构对齐公开 0.70 的 CropUNet：resnet18 encoder + 5 级 U-Net decoder，3 通道输入 `[gray, CLAHE, seed]`，输出 1 通道概率。
- crop：以 seed mask 的 bbox 取方形（context=1.8×max(side)，min 96px），从 **原始 2048 分辨率**裁出后 resize 到 256。
- 训练 seed：GT mask 做腐蚀/膨胀(±6)、平移(±12%)、粗化(2-4x 下采样再上采样，模拟 YOLO 低分辨率 mask)、或填充 box seed 的混合；
  另有 rot90/flip/zoom(±15%)/gain 增强。
- 训练目标：该 annotator 记录的实例 GT 掩膜；loss = 0.45·BCE + 0.55·soft Dice；AdamW lr=3e-4, cosine, 8 epochs, batch 24。
- crop cache 用 uint8 memmap 预计算（`work/yolo_data/refiner_cache/fold_0/all_c256/`），每 epoch 只需几秒。

## 3. 推理与 PQ 后处理
1. YOLO11-seg 单次前向（imgsz=1024, conf=0.05 低阈值取候选），实例 mask 去重（IoU>0.6 保留高分）。
2. 每个 seed → 2048 全分辨率 crop → refiner 4-flip TTA（identity/hflip/vflip/hvflip）+ 拓扑守卫
   （仅当面积比 0.60-1.70、Dice≥0.62、连通域数不增才接受 TTA），refiner 阈值扫描。
3. 逐实例概率 resize 回 bbox 尺寸后再阈值（不在 256 网格上硬阈值，边界更准）。
4. 后处理扫描维度（PQ-aware，fold0 val 直接最大化官方 pooled PQ）：
   detector conf ∈ {0.05,0.10,0.15,0.20,0.25,0.30}，refiner 阈值 ∈ {0.45,0.55,0.65,0.75}，
   形态学 ∈ {none, dilate1, erode1}（±1px @2048），min_area ∈ {20,50,96}，
   重叠消解顺序 ∈ {det×refiner conf, detector conf}。
5. 重叠消解：按分数降序贪心 painting，保证**同一图任意两个 mask 像素不重叠**（官方硬约束）。
6. 每实例输出 confidence = detector_conf × refiner_conf，落盘 `*_instances.csv` 供后续与公开 0.70 集成。
- 所有中间产物（det score、crop bounds、refiner 概率图、seed mask）缓存到 `work/models/preds/<split>/*.npz`，
  阈值/形态学扫描纯 CPU 进行，不重跑 GPU。

## 4. 环境问题（重要）
- 训练期间发现 GPU 被锁 30W 功耗墙（SM 300MHz, Default 90W）→ 正常算力 1/7；用户解除后恢复 90W/P0（SM 1100-1800MHz）。
- 满速实测（RTX 2080 Super Max-Q 8GB）：n-seg@1024 batch8 ≈ 1.0-1.3 s/it（~6-8 img/s）；s-seg@1024 batch8 ≈ 3.5 s/it（~2.3 img/s，显存 8.2GB 到顶）。

## 5. 结果

### 5.1 云端 T4 全量训练（主线模型，task-9）
- Kaggle kernel `ptcuoigl/filament-cloud-runner-v1` **COMPLETE，总耗时 5311s ≈ 89 min**
  （detector 3263s / refiner 554s / val 推理+sweep 1370s / test 90s）。
- 训练配置：yolo11s-seg @1024, batch 8, **30 epochs**, all-annotations（fold0 训练集 565 物理图 / 922 标注记录；
  val 142 图 / 232 annotator-images）；refiner ResNet18 U-Net 8 epochs（crop 256, batch 16）。
- **fold0 val 官方 pooled PQ = 0.378049**（SQ 0.647527 / RQ 0.583836；TP 968 / FP 531 / FN 849；n_gt 1817；
  最优配置 n_pred 919）。诚实 OOF：fold0 的 142 张物理图完全没参与训练。
- 最优配置：detector conf=0.35，refiner 阈值=0.50，4-flip TTA=guard，img_size=1024，min_area=48，逐图贪心 painting。
- 产物：work/cloud/train_out/{best_det.pt, best_ref.pt, submission.csv, val_report.json, sweep.csv, val_per_image.csv}。

### 5.2 本地重算 + 加密阈值扫描
- 用云端权重本地重跑同一管线：**PQ/SQ/RQ/TP/FP/FN 与云端逐位一致（0.378049447467445）** → 导出管线可信。
- 本地加密扫描（90 组：conf∈{0.30..0.55} × refiner 阈值∈{0.40..0.60} × min_area∈{20,48,96}，
  work/models/sweep_cloud_val.csv / .json，top 配置用 src/pq_metric.py 复核）：
  **新最优 = conf 0.35 / refiner 阈值 0.40 / min_area 96 / tta guard → 官方 pooled PQ = 0.380312**
  （SQ 0.647902 / RQ 0.586989；TP 970 / FP 518 / FN 847；n_pred 913），较云端 0.378049 提升 +0.0023
  （FP 531→518；min_area 48→96 去掉小碎片）。云端 sweep 的 conf 上界 0.35 确实落在边界，但更高 conf 反而略降。
- 预测统计（val）：919 实例 / 139 图（142 张 val 图里 3 张无预测）；test：1065 实例 / 177 图。

### 5.3 与参照对比
| 方案 | 指标口径 | pooled PQ | 备注 |
|---|---|---|---|
| 人类标注者互评 | PQ | 0.3399 | harness：单标注者 vs 其他标注者（31 位, 1196 annotator-images） |
| 多数投票共识掩膜（oracle 上界，非可提交） | PQ | 0.6547 | harness：296 张多标注图 |
| 公开 LB 0.70 方案（旧 Dice 时代） | PQ | 0.34 | 主办方改指标后同一 pipeline 复现 |
| **本方案（自训 YOLO11s-seg + refiner，云端 T4）** | **PQ** | **0.3780** | fold0 诚实 OOF |

### 5.4 精修 dariush 掩膜实验（负结果，有价值）
用本方案训练好的 crop refiner（work/cloud/train_out/best_ref.pt）精修 dariush fold0 OOF 掩膜
（142 图 / 1076 掩膜，baseline 官方 pooled PQ = 0.475472 / SQ 0.688052 / RQ 0.691041；
TP 1238 / FP 528 / FN 579），4-flip 拓扑守卫 TTA + 阈值 0.30-0.60 + 4 种替换策略。

| 变体 | 阈值 | 官方 PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| 不精修（baseline） | - | **0.475472** | 0.688052 | 0.691041 | 1238 | 528 | 579 |
| 只替换面积变化<30% | 0.30 | 0.446087 | 0.665971 | 0.669830 | 1200 | 566 | 617 |
| 全替换 | 0.30 | 0.445932 | 0.665739 | 0.669830 | 1200 | 566 | 617 |
| dice>0.6 / >0.7 才替换 | 0.30 | 0.445932 | 0.665739 | 0.669830 | 1200 | 566 | 617 |

结论：**精修一律变差（-0.029 PQ）**。refiner 输出与 dariush 掩膜平均 dice=0.923、面积比 0.972
（系统性缩小 2.8%），导致 TP 1238→1200、FN 579→617：恰好把 dariush 的"IoU 0.25-0.5 近失"样本推到 0.5 以下，
SQ 也从 0.688→0.666。dariush 的 10×U-Net 集成（模型平均≈标注者共识）已经比我们单模型 refiner 更贴近 pooled GT，
**结论：不要把本 refiner 盲用于更好的外部掩膜；我们的 refiner 只适合为自己的 detector seed 服务**（其自身 pipeline SQ 0.648）。
（实验脚本：src/refine_dariush.py；结果 work/models/refine_dariush_fold0/results.{csv,json}，PEAK_MEMORY 0.223 GB。）

### 5.5 消融：detector-only（no_refiner）vs 完整 pipeline（fold0 val, 官方 pooled PQ）
| 配置 | PQ | SQ | RQ | TP | FP | FN | n_pred |
|---|---|---|---|---|---|---|---|
| no_refiner, conf=0.30（最优） | 0.356932 | 0.638195 | 0.559283 | 967 | 674 | 850 | 1004 |
| no_refiner, conf=0.35（与完整管线同 conf） | 0.355026 | 0.640017 | 0.554714 | 915 | 567 | 902 | 909 |
| **完整 pipeline**（conf=0.35 / ref=0.40 / min_area=96） | **0.380312** | 0.647902 | 0.586989 | 970 | 518 | 847 | 913 |
→ refiner 贡献：同 conf 下 **PQ +0.0253**（SQ +0.008, RQ +0.032, TP +55, FP -49）；PEAK_MEMORY 0.438 GB。
明细 work/models/ablation_no_refiner.json（10 个 conf 全表，官方 pq_metric）。

### 5.6 semantic U-Net 共识软标签模型评估与集成实验（cloud）
云端 `src/train_semantic.py`（efficientnetv2_s UNet, 共识软标签, 1 通道, 2048 全卷积推理, KEEP 规则出实例）：
| 折 | 默认 KEEP (0.35/0.99/200/0.85) | 调优 KEEP | 提升 |
|---|---|---|---|
| fold0（2ep smoke） | PQ 0.345821（SQ .651028/RQ .531193, TP1026/FP1020/FN791） | **0.353007**（t_lo.35/t_hi.95/min_area 400/meanp.85; SQ .651373/RQ .541943, TP982/FP825/FN835） | +0.0072 |
| fold1（2ep smoke） | PQ 0.349136（SQ .671389/RQ .520021, TP974/FP1133/FN665） | **0.366857**（t_lo.30/t_hi.95/min_area 400/meanp.85; SQ .673570/RQ .544646, TP918/FP814/FN721） | +0.0177 |
- 默认 KEEP 与云端 val_report.json 逐位一致（prob_ensemble 复核）；`min_area 200→400` 是稳定增益（去掉小碎片、降 FP）。
- **用我们的 refiner 精修 semantic 掩膜 = 小负收益**：baseline 0.353592 → 最优变体 0.347766（-0.0058）。
  对比精修 dariush（-0.029）：外部掩膜质量越好，我们的 refiner 伤害越大；refiner 只适合服务自己的 detector seed。
- 概率级融合工具就绪：`src/fuse_probs_sweep.py`（多 dump 加权 mean/wmean/max/median + KEEP + 官方 PQ；
  支持每 dump 独立权重组合、**共识目标 PQ'**）；`src/eval_semantic.py`（semantic val_probs 的 KEEP 扫描 + 官方复核）。
  验证：①dariush TTA8 概率 dump 自融合 → PQ=0.475472 逐位复现 ✓；②dariush test probs → submission 1177 实例/177 图，
  与 submissions/dariush_unet_full_v1.csv 行数/ID 完全一致（1075/1177 RLE 字节相同，其余为同实例的边界级差异）✓。
- semantic test 推理：`src/infer_semantic_test.py` 支持多 ckpt 概率平均；全卷积 2048 会超出 8GB（WDDM 溢出到系统内存，
  PEAK 16.9GB、~28s/图），**tile=1024 重叠拼接**版 PEAK 4.39GB、~5s/图（且 1024 与训练 crop 一致）。
- 云端 full semantic（24ep, folds 0/1, 双 T4）完成后按同样流程评估 + 与 dariush 概率融合（最有希望超过 0.4755 的路径）。

### 5.7 关键结论与教训
- PQ 指标下 SQ 已接近人类互评水平（0.648 vs 0.634），主要损失来自 RQ（FP/FN，0.584）——漏检与多余实例各罚 0.5。
- 阈值对 PQ 极敏感但平台较平：conf 0.30→0.35 只差 ~0.001 PQ；refiner 阈值 0.50 优于 0.60/0.65（0.378 vs 0.374/0.373）。
- **不要做形态学膨胀/腐蚀**（harness 实验：共识掩膜 dilate1 使 PQ 0.655→0.537）——让 refiner 阈值决定掩膜粗细。
- 多模型/多折概率平均是逼近标注者共识的关键手段（时间未允许，后续工作）；refiner 输出概率逐像素平均比 mask 投票更平滑。
- 算力教训：本地 GPU 曾被 30W 功耗墙锁在 300MHz（s-seg@1024 需 3.4h）；解墙后 90W/P0（n-seg@1024 ≈100s/epoch）；
  云端 T4 才是主线（s-seg 30ep 仅 54min），本地用于导出/验证。

## 5b. 交付物清单
- 模型：work/cloud/train_out/best_det.pt（yolo11s-seg）、best_ref.pt（ResNet18 U-Net refiner）；
  本地备份：work/models/yolo/fold_0_all/best.pt（yolo11n-seg 20ep）、work/models/refiner/fold_0_all/best.pt（12ep 中 best）
- 预测/提交：submissions/ours_model_v1.csv（**1052 实例/177 图**，最佳配置 conf .35/ref .40/min_area 96，validate OK；
  若需 0.378 配置版可从 work/models/test_pred_instances.csv 的早期版本重导）、
  work/models/fold0_val_pred.csv（919 实例/139 图，带 confidence）、work/models/test_pred_instances.csv
- 分析实验：work/models/sweep_cloud_val.{csv,json}（阈值/形态学/min_area 扫描）、
  work/models/ablation_no_refiner.json（no_refiner 消融）、work/models/refine_dariush_fold0/（精修 dariush 负结果）、
  work/models/refine_sem_fold0/（精修 semantic 负结果）、work/models/sem_fold0_eval + sem_fold1_eval（semantic smoke 评估）
- 脚本：src/train_yolo_seg.py（数据/检测器）、src/train_refiner.py（refiner）、src/infer_modeling.py（推理/PQ 扫描）、
  src/cloud_pipeline.py（自包含云端训练+推理）、src/export_preds.py（云端权重导出带 confidence）、src/sweep_cloud_val.py（加密扫描）
- 云端：src/cloud_pipeline.py 由 kaggle-runner 打包成 kernel（SMOKE 验证 → 全量 89min）。

## 6. 复现命令
```bash
# 数据准备（折分/medoid/全标注数据集）
python src/train_yolo_seg.py data --folds 0 --label-source all --resize 1024
# 检测器
python src/train_yolo_seg.py train --fold 0 --label-source all --model yolo11n-seg.pt --epochs 20 --batch 8 --imgsz 1024 --workers 8 --no-val --cache disk
# refiner crop cache + 训练
python src/train_refiner.py cache --fold 0 --label-source all
python src/train_refiner.py train --fold 0 --label-source all --epochs 8 --batch 24
# 推理缓存 / 阈值扫描 / 官方 PQ / 提交
python src/infer_modeling.py cache-preds --fold 0 --label-source all
python src/infer_modeling.py sweep --fold 0 --label-source all
python src/infer_modeling.py evaluate --fold 0 --label-source all --conf <best> --ref <best> --morph <best> --min-area <best>
python src/infer_modeling.py cache-preds --test --label-source all
python src/infer_modeling.py predict-test --fold 0 --label-source all --conf <best> --ref <best>
python src/validate_submission.py <submission.csv>
# 云端（Kaggle T4, kaggle-runner 执行）
python src/cloud_pipeline.py --smoke   # 首次验证
python src/cloud_pipeline.py           # 全量 yolo11s-seg 30ep + refiner 8ep + fold0 val PQ + submission
```

## 2026-10-29 鏅氶棿杩藉姞锛坰em 澶氭姌 + notoverkil 澶嶇幇锛?
### semantic U-Net锛坋ncoder tu-tf_efficientnetv2_s, 1ch, tile 1024 鎺ㄧ悊锛?| 妯″瀷 | 榛樿 KEEP PQ | 鏈€浼?KEEP PQ | 鏈€浼?KEEP |
|---|---|---|---|
| fold0 bcedice 24ep (v2) | 0.421075 | 0.428201 | t_lo .35 / t_hi .95 / area 200 / min_meanp .90 |
| fold0 bcedice_cldice 16ep (v3) | 0.411483 | 0.411483 | 榛樿鍗虫渶浼?-> clDice 涓嶅 bcedice |
| fold1 bcedice 24ep (v2) | 0.428887 | 0.442906 | t_lo .40 / t_hi .95 / area 400 / min_meanp .90 |
| fold2 bcedice 24ep (v3) | 0.426439 | 0.431197 | t_lo .40 / t_hi .95 / area 200 / min_meanp .90 |

瑙勫緥锛歮in_meanp 0.85->0.90 + t_hi 0.95 鏄ǔ瀹氬鐩婏紙鍘绘帀浣庣疆淇″疄渚嬶紝FP 澶ч檷锛夈€?鍗曟姌 dariush x sem(fold0) 铻嶅悎鍏?w 鍧囦綆浜?dariush 鍗曟簮 0.4755锛堥粯璁?KEEP best 0.4721@w=.1锛?sem 鍗曟姌鏈€浼?KEEP 鍙嶈€屾洿宸?0.4534@w=.15锛?> 鍗曟姌铻嶅悎 gate FAIL锛屼富绾挎敼涓哄鎶樺钩鍧囥€?
### notoverkil锛坧ublic LB 0.41锛?5 妯″瀷 5 鏃忥級澶嶇幇
- 浜戠 T4 fold0 鍗曡鍥撅細maskrcnn1536 PQ 0.2589锛堣鏍间笉绗︼級銆亂olo11m1536 PQ 0.4357銆?  mask-level 铻嶅悎 0.4047銆?- 8-view dihedral TTA + 鏃忓唴 consensus锛堟湰鍦?CPU锛岀敤浜戠 kernel 鐨?cache锛夛細
  yolo11m-1536 f0 PQ 0.441490锛圫Q .6835/RQ .6459锛孴P1351/FP1015/FN466锛夆増 鍗曡鍥?+0.006銆?- 鑴氭湰锛歴rc/cloud_notoverkil_ensemble.py锛坒old0-eval / test / assemble + shard锛沜onsensus 鏀?bbox 灞€閮ㄦ姇绁紝
  test 闃舵 4000 瀹炰緥/鍥惧繀椤荤敤鏂扮増锛夛紱custom-format yolo ckpt 閲嶅缓閫昏緫鍦?src/infer_notoverkil.py銆?- 铻嶅悎瀵圭収鑴氭湰锛歴rc/fuse_ensemble_with_dariush.py锛? 绛栫暐锛夈€佹牎楠?src/check_ensemble_candidate.py銆?- kernel rc=1 鏍瑰洜锛歝onvnext 鏃忕己 segmentation_models_pytorch -> 绌?cache -> 绌?pred 璇勪及鎶?ValueError锛涘凡鍔犵┖鏃?skip 淇濇姢銆?
### 浜戠 kernel 鍖?- work/cloud/kernel_sem_test_probs/锛歴em 5 鎶?test probs锛圱4x2锛屾寜鍥剧墖鎷?shard锛屽妯″瀷骞冲潎锛岃緭鍑?layout a npz锛夈€?- 鏉冮噸 dataset 鎵撳寘鐩綍锛欳:\Users\ptcu_\filament_assets\weights\sem_weights\锛坒old0/1/2 bcedice + fold0 cldice锛宖old3/4 寰呰ˉ锛夈€?- 鍛戒护娓呭崟锛歯otes/FUSION_COMMANDS.md銆?

### 5 鎶?sem 瀹屾暣琛?+ 铻嶅悎棰勬湡锛?026-10-30 00:45锛?| fold | 榛樿 KEEP PQ | 鏈€浼?KEEP PQ | 鏈€浼?KEEP |
|---|---|---|---|
| 0 | 0.421075 | 0.428201 | .35/.95/200/.90 |
| 1 | 0.428887 | 0.442906 | .40/.95/400/.90 |
| 2 | 0.426439 | 0.431197 | .40/.95/200/.90 |
| 3 | 0.434233 | 0.436190 | .30/.95/200/.85 |
| 4 | 0.407385 | 0.412258 | .30/.99/400/.85 |
| 鍧囧€?| 0.4236 | 0.4302 | 鈥?|

- fold0 铻嶅悎鏍″噯锛坉efault KEEP锛夛細dariush 0.475472锛泈=0.05 -> 0.474640锛?.075 -> 0.473190锛?.10 -> 0.473086锛?  0.125 -> 0.472567锛?.15 -> 0.469668锛?.20 -> 0.468036锛?.30 -> 0.467230锛坵 瓒婂皬瓒婂ソ锛屽崟鎶?sem 琛ヤ笉鍔?dariush锛夈€?- 5 妯″瀷 test 骞冲潎棰勬湡 0.440-0.450锛涗笌 dariush 铻嶅悎棰勬湡 0.475-0.481 @ w=0.05-0.10锛沢ate 0.4785 杈归檯銆?- 鎺╄啘绾ц瀺鍚堬紙sem instances + dariush锛夊叏閮ㄤ綆浜庡熀绾匡紙recover 0.4508/0.4584锛宑onsensus 0.4536锛?> 鍚︽帀锛屽彧鐢ㄦ鐜囩骇 wmean銆?- sem test kernel 淇锛氱己 smp 闇€ pip 鑷妇锛涙帹鐞嗛渶 no_grad锛堝惁鍒?numpy() on graph tensor锛夈€?

### sem 澶氭姌铻嶅悎璺嚎鏈€缁堢粨璁猴紙2026-10-30 01:55锛?- 5 鎶樺钩鍧囦箰瑙備唬鐞嗭紙crossfold probs锛宖old1-4 娉勬紡锛夛細sem_avg default KEEP PQ 0.444637锛圥Q' .3901锛夛紱
  鍗曟ā娉勬紡 未=+0.0081 鈫?鍘诲亸璇氬疄浼拌 鈮?0.4365锛堜笌鍗曟姌 KEEP 鏈€浼樺潎鍊?0.4302 閲忕骇涓€鑷达級銆?- dariush 脳 sem_avg 铻嶅悎 proxy锛坉efault KEEP锛夛細w=.05鈫?.474857锛寃=.075鈫?.475063锛堟渶浼橈級锛寃=.10鈫?.474729锛?  w=.15鈫?.472630锛寃=.20鈫?.472344锛寃=.30鈫?.469058锛涘鐓?dariush only 0.475472銆?- **鍗充娇 leaky 涔愯鍙ｅ緞铻嶅悎浠嶄綆浜?dariush锛?0.0004锛夛紝gate 0.4785 涓嶅彲杈?鈫?澶氭姌 sem 璺嚎鏀惧純锛屾渶缁堟彁浜や繚鎸?dariush銆?*
- harness 鐙珛澶嶆牳锛氭暟瀛椾竴鑷达紙leak 未 +0.0081銆佸幓鍋?0.4365銆乫used w=.05鈫?4744/w=.10鈫?4738锛夛紱sem KEEP 璺ㄦ姌杩佺Щ鎬?  fold0/fold1 浜掓崲浠?-0.003~-0.006锛宮mp .90 绋冲畾锛涙姌涓?KEEP 寤鸿 t_lo .375/area300/mmp .90銆?- 鍊欓€夋枃浠朵繚鐣欏湪 submissions/candidates/锛坒used_w005..w050銆乻em_only锛屽叏閮?validate=0锛夛紝浠呬綔璁板綍/鍚庣画鍙傝€冦€?- nk锛坣otoverkil锛夊鐜扮粨璁猴細鍗曟棌 8TTA+consensus 0.4415锛宑ross-family 0.4134锛堚増鍏?LB 0.41锛夛紝maskrcnn/convnext 鏈鐜帮紱
  寮变簬 dariush锛屼粎浣滅浜屾柟娉曡褰曘€?

### nk v3 test锛坣otoverkil 澶嶇幇锛屾渶缁堣褰?2026-10-30 03:30锛?- 浜戠鍙屽崱 25 妯″瀷 脳 8 瑙嗗浘 脳 180 鍥撅紝3h49m 瀹屾垚锛涘疄闄呬骇鍑?15 涓?yolo 妯″瀷锛? 鏃徝? 鎶橈級consensus锛坈onvnext/maskrcnn 鏈窇锛夈€?- submission锛?328 琛?/ 180 鍥撅紙骞冲潎 12.9 瀹炰緥/鍥撅紝闈㈢Н涓綅 975锛夛紱schema/瑕嗙洊/閫愬浘闆堕噸鍙犳牎楠?PASS銆?- 涓?dariush 瀵圭収锛歞ariush 瀹炰緥 90.1%锛?061/1177锛夎 nk 鍖归厤锛圛oU>0.5锛夛紝浣?nk 棰濆 1251 瀹炰緥锛?4%锛夛紝楂樼簿搴﹀尮閰嶄粎 216 鈫?纰庣墖鍖?澶氭鏄庢樉锛屼笌鍏?OOF锛坈ross-family 0.4134锛孎P 1372锛変竴鑷淬€?- 缁撹锛氬悎娉曞弬鑰冨€欓€夛紝寮变簬 dariush锛圤OF 鍙ｅ緞锛夛紝涓嶅崰鎻愪氦閰嶉锛涙枃浠?C:\...\downloads\nk_test\submission.csv銆丏: submissions/candidates/nk_ensemble_v3.csv锛屾牎楠?work/models/nk_candidate_check.json銆?
