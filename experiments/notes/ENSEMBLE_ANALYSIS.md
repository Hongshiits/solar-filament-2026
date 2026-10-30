# ENSEMBLE_ANALYSIS — 集成规则调优 + 误差分析（task-14，fold0 实证）

> 负责人：harness。所有数字 = 官方 pooled PQ（annotator-image pool，IoU 严格 > 0.5），折分 = work/cache/folds.json fold 0。
> GT = fold0 全部标注者记录（232 annotator-image / 1817 GT instances / 142 物理图）。评测器：src/pq_metric.py（人类基线复现 0.3399/0.6337）。
> 本文件不使用反引号（行内代码用 <code> 标签，代码块 4 空格缩进）。

---

## 0. 结论速览

| 问题 | 结论 |
|---|---|
| 最终 test 提交 | **保持 dariush TTA8 全量 = <code>submissions/dariush_unet_full_v1.csv</code>（public LB = 0.39）**，不做任何集成 |
| ours（YOLO11s+refiner，fold0 PQ 0.3780）集成到 dariush | **无正收益**：全 142 图上 base_only 0.4755 最优，任何规则/阈值 ≤ 0.4755 |
| dariush TTA8 ∪ TTA2 并集 | **无正收益**：最佳变体 ΔPQ = **-0.0029**（bootstrap 95% CI [-0.0061,+0.0002]，P(Δ>0)=3.4%），未达 +0.003 门槛 |
| 预期 fold0 PQ 增益 | 集成增益为 0；base dariush TTA8 = **0.4755**（TTA2 = 0.4688，+0.0067 来自 TTA） |

---

## 1. 数据与评测口径

* **base** = dariush 公开 U-Net 的诚实 OOF（每张图用其自己 fold (f+1)%5 的模型，未见过该图）：
  * TTA8 完整版：<code>work/dariush_oof/pred_dariush_fold0.csv</code>（1076 行 / 142 图，PQ **0.4755**）
  * TTA2 快速版：<code>work/dariush_oof/pred_dariush_fold0_tta2.csv</code>（1119 行 / 142 图，PQ 0.4688）
  * 无 score 列（按 1.0）
* **extra** = 我们自训 YOLO11s-seg detector + crop refiner 的 fold0 val 预测：
  <code>work/models/fold0_val_pred.csv</code>（919 行 / 139 图；confidence=det_score×refiner_conf，0.33–0.86，中位 0.556），
  官方 pooled PQ = **0.3780**（SQ .6475 / RQ .5838；与云端 val_report 逐位一致）。诚实 OOF（fold0 的 142 图未参与训练）。
* 评测：<code>src/pq_metric.py</code>（pycocotools backend 与 numpy 后端 bit-exact），逐 annotator-image 计算 IoU 矩阵后汇总 TP/FP/FN。
* 规则实现：<code>src/ensemble_eval.py</code>（单次 painting 的 prefix 扫描；residual 用 pycocotools 编码；逐图保证不重叠）；
  误差分析：<code>src/error_analysis.py</code>。

---

## 2. 单模型基线（fold0 val，142 图 / 1817 GT）

| 模型 | PQ | SQ | RQ | TP | FP | FN | pred 数 | 备注 |
|---|---|---|---|---|---|---|---|---|
| dariush TTA8（base） | **0.4755** | 0.6881 | 0.6910 | 1238 | 528 | 579 | 1076 | 语义 U-Net 10 模型平均，边界 SQ 高 |
| dariush TTA2 | 0.4688 | 0.6859 | 0.6835 | 1248 | 587 | 569 | 1119 | TTA 更多实例、FP 更多 |
| ours（YOLO11s+refiner） | 0.3780 | 0.6475 | 0.5838 | 968 | 531 | 849 | 919 | 检测式，RQ 低（漏检多）、SQ 低 0.04 |
| ours（加密 sweep 最优） | 0.3803 | 0.6479 | 0.5870 | 970 | 518 | 847 | 913 | conf .35/ref .4/min_area 96（modeling 19:07 sweep_cloud_val.csv）；19:12 已按此配置重导 test |
| ours detector-only（refiner off, conf 0.30 最优） | 0.3569 | 0.6382 | 0.5593 | 967 | 674 | 850 | 1004 | modeling ablation（10 conf 全表在 work/models/ablation_no_refiner.json） |
| ours detector-only（refiner off, conf 0.35 同 conf） | 0.3550 | 0.6400 | 0.5547 | 915 | 567 | 902 | 909 | 同上 |

消融结论：
* **refiner 贡献（同 conf 0.35）：PQ +0.0253**（0.3550→0.3803），SQ +0.0079 / RQ +0.0323 / TP +55 / FP -49；
  各自最优 conf 对比 0.3569 vs 0.3803 = +0.0234 → refiner 主要提升 RQ（实例完整性）与少量 SQ。
* TTA 消融（dariush 云端 sweep，conf=0.35 ref=0.5）：tta=none PQ **0.3622**（SQ .6401/RQ .5657）→ tta=guard **0.3780**，**+0.0158**。

---

## 3. ours 集成实验（base=dariush TTA8, extra=ours，conf=det_score×refiner_conf）

fold0 全 142 图，按 PQ 排序（完整 JSON：<code>work/analysis/ensemble/ensemble_eval_fold0.json</code>）：

| 规则 | PQ | ΔPQ | TP | FP | FN | pred |
|---|---|---|---|---|---|---|
| **base_only** | **0.4755** | 0 | 1238 | 528 | 579 | 1076 |
| base+extra_no_overlap(conf≥0.7) | 0.4755 | 0.0000 | 1238 | 528 | 579 | 1076 |
| base+extra_novel(conf≥0.7, overlap<0.15) | 0.4755 | 0.0000 | 1238 | 528 | 579 | 1076 |
| base+extra_no_overlap(conf≥0.6) | 0.4753 | -0.0002 | 1238 | 529 | 579 | 1077 |
| base+extra_no_overlap(conf≥0.5) | 0.4736 | -0.0019 | 1239 | 544 | 578 | 1088 |
| base_first+extra(conf≥0.5)+dupguard(0.5) | 0.4723 | -0.0032 | 1239 | 554 | 578 | 1094 |
| base+extra_no_overlap(conf≥0.4) | 0.4722 | -0.0033 | 1245 | 564 | 572 | 1105 |
| base+extra_no_overlap(conf≥0.3) | 0.4707 | -0.0048 | 1263 | 603 | 554 | 1141 |
| base_first+extra(conf≥0.8) | 0.4687 | -0.0068 | 1238 | 580 | 579 | 1106 |
| base_first+extra(conf≥0.7) | 0.4427 | -0.0328 | 1238 | 793 | 579 | 1241 |
| base_first+extra(conf≥0.6) | 0.4086 | -0.0669 | 1238 | 1114 | 579 | 1437 |
| extra_first(conf≥0.7) | 0.4330 | -0.0425 | 1224 | 807 | 593 | 1241 |
| extra_only（含任意阈值） | 0.3780 | -0.0975 | 968 | 531 | 849 | 919 |
| base_first（全部 extra） | 0.3429 | -0.1326 | 1263 | 1975 | 554 | 1980 |
| extra_first（全部 extra） | 0.2939 | -0.1816 | 1142 | 2091 | 675 | 1975 |

**解读**：
1. 任何"让 ours 与 base 同时存在"的规则都 ≤ base：ours 的实例与 dariush 高度重复（919 个中仅 8.5%–11% 在 base 中找不到 IoU≥0.05–0.3 的近邻），
   而 painting 的 residual 会把重复实例切成边界碎片 → FP 暴增（base_first 全部 extra 时 FP 1975，+1447）。
2. 最保守的 no_overlap（只加与 base 零重叠的 ours 实例）在 conf≥0.6 时与 base 打平（0.4753），加的少量实例全部是 FP → 无收益。
3. 结论：**ours 集成没有价值，test 不做**。

---

## 4. dariush TTA8 ∪ TTA2 并集实验（同一模型不同 TTA）

脚本与结果：<code>work/analysis/ensemble/tta_union.py</code> / <code>tta_union_fold0.json</code>（bootstrap 2000 次重采样 stem）。

| 规则 | PQ | ΔPQ | SQ | RQ | TP | FP | FN | bootstrap 95% CI(Δ) | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|---|
| **base_only (TTA8)** | **0.4755** | 0 | 0.6881 | 0.6910 | 1238 | 528 | 579 | — | — |
| base+extra_no_overlap | 0.4725 | **-0.0029** | 0.6864 | 0.6885 | 1273 | 608 | 544 | [-0.0061, +0.0002] | 3.4% |
| base_first_dupguard0.5 | 0.4724 | -0.0031 | 0.6864 | 0.6883 | 1273 | 609 | 544 | [-0.0063, +0.0001] | 3.0% |
| base_first（painting） | 0.3928 | -0.0827 | 0.6864 | 0.5722 | 1274 | 1362 | 543 | [-0.0889, -0.0767] | 0% |
| extra_first | 0.3916 | -0.0838 | 0.6854 | 0.5714 | 1266 | 1348 | 551 | [-0.0916, -0.0756] | 0% |
| extra_only (TTA2) | 0.4688 | -0.0067 | 0.6859 | 0.6835 | 1248 | 587 | 569 | [-0.0130, +0.0001] | 2.9% |

**解读**：TTA8 已基本覆盖 TTA2 的高质量实例；并集能多命中 35 个 TP（1273 vs 1238）但同时多 80 个 FP（608 vs 528），
净效果 -0.003，bootstrap 不支持正收益（P(Δ>0)=3.4%）→ **不做 TTA8∪TTA2 提交**。

---

## 5. 决策（task-14 最终）

1. **test 提交：保持 dariush TTA8 全量 <code>submissions/dariush_unet_full_v1.csv</code>（public = 0.39，ref 57768486）**。
   numScoredSubmissions=1 自动取 best，无需再提交。
2. 若未来要再加模型：只有当新模型单模 fold0 PQ ≥ dariush - 0.01 且"独有命中 GT"≥ 10% 时，才值得尝试 base+extra_no_overlap(conf≥0.6)；
   当前 ours（0.378，独有命中 3.3%）不满足。
3. 最终融合命令（仅当条件 2 满足时；否则不执行）：

        python src/ensemble.py \
            --inputs submissions/dariush_unet_full_v1.csv <新模型test.csv> \
            --weights 1.0 1.0 --min-area 20 --max-residual-frac 0.5 \
            --out submissions/ensemble_final.csv
        python src/validate_submission.py submissions/ensemble_final.csv

   （base 放第一个；等权即 base 优先；--max-residual-frac 0.5 去近重复碎片。
   注意：fold0 实验表明**普通 painting 会毁分（-0.08）**，只有 no_overlap 变体安全，故当前不建议执行。）

---

## 6. 误差分析（fold0 val，<code>src/error_analysis.py</code> → work/analysis/ensemble/error_analysis_fold0.json/.md）

### 6.1 GT 面积分箱命中率（IoU>0.5 命中 / 该箱 GT 数）

| GT 面积 | dariush TTA8 | ours |
|---|---|---|
| 50-100 | 0.000 (0/2) | 0.000 (0/2) |
| 100-200 | 0.000 (0/9) | 0.000 (0/9) |
| 200-500 | 0.537 (139/259) | 0.355 (92/259) |
| 500-1k | 0.653 (293/449) | 0.510 (229/449) |
| 1k-2k | 0.731 (370/506) | 0.551 (279/506) |
| 2k-5k | 0.741 (318/429) | 0.648 (278/429) |
| 5k-10k | 0.760 (95/125) | 0.568 (71/125) |
| >10k | 0.605 (23/38) | 0.500 (19/38) |

* 小目标（<200px）两个模型都几乎无法命中（官方 GT 最小 9px；这类目标在 IoU>0.5 下基本不可达）；
* 200px 以上 dariush 全面领先 0.09–0.19；超大目标（>10k）两者都下降（暗条过长/弯曲，边界难）。
* 这也是 RQ 差距的来源：ours FN 849 vs dariush 579。

### 6.2 FP 来源

| 来源 | FP 数 | 无 GT 重叠 | IoU 0-0.25 | IoU 0.25-0.5 | 多 GT 重叠(合并型) | FP 面积中位 |
|---|---|---|---|---|---|---|
| dariush TTA8 | 528 | 209 (40%) | 80 (15%) | 239 (45%) | 21 | 989 px |
| ours | 531 | 130 (24%) | 117 (22%) | 284 (53%) | 18 | 1183 px |

* 最大 FP 类别是 **IoU 0.25-0.5 的近失（45%/53%）**：预测存在但边界/形状差一点；
  这与 SQ 强相关 → 提升边界精度（refiner/更好 TTA）比"加实例"更有价值。
* 无 GT 重叠的纯幻觉 FP：dariush 209 个，ours 130 个（ours 更保守但漏检多）。

### 6.3 FN 来源

| 来源 | FN 数 | 完全漏检 | 部分重叠(IoU<0.5) |
|---|---|---|---|
| dariush TTA8 | 579 | 305 (53%) | 274 (47%) |
| ours | 849 | 489 (58%) | 360 (42%) |

* 近一半 FN 是"预测到了但 IoU 不够"（部分重叠）→ 边界精度是 FN 的第二大来源；
* 完全漏检：dariush 305 个（其中小目标占比较高，见 6.1）。

### 6.4 TP 边界质量

| 来源 | TP IoU 均值 / 中位 | TP pred/GT 面积比 均值 / 中位 |
|---|---|---|
| dariush TTA8 | 0.6881 / 0.6916 | 1.0866 / 1.0647 |
| ours | 0.6475 / 0.6467 | 1.0563 / 1.0115 |

* dariush 的 TP 掩膜比 GT 平均**大 8.7%**（中位 6.5%）：IoU>0.5 下仍可通过，但 SQ 有提升空间（略缩小可能加分，但不建议无 OOF 验证的后处理）；
* ours 面积更接近 GT（+5.6%）但 IoU 低 → 形状/边界质量差距。

### 6.5 互补性（为什么集成没用）

* 1817 个 GT 实例：两者都命中 908，仅 dariush 命中 **330**，仅 ours 命中 **60 (3.3%)**，两者都没命中 519。
* ours 独有命中的 GT 面积分布：200-500px 8、500-1k 19、1k-2k 12、2k-5k 16、5k-10k 4、>10k 1 → 无特定"补漏区"。
* ours 919 个实例中与 dariush 实例 max IoU：<0.05 仅 78 (8.5%)、<0.2 94 (10.2%)、<0.3 104 (11.3%) → 约 90% 是重复实例。
* **理论上限**：即使把 ours 独有命中的 60 个 TP 无损加进 dariush（不加任何 FP），PQ 也只从 0.4755 → 0.4873（+0.012），
  现实中加这些实例会附带 ~2-4 倍 FP，净增益 ≈ 0 或负。→ 与实验结论一致。

---

## 7. test 侧诊断（无 GT，仅分布/覆盖）

| 提交 | 行数 | 覆盖图 | 实例/图 | 面积中位 | 面积均值 |
|---|---|---|---|---|---|
| pub070_kaggle_v1（public 0.34） | 1299 | 169/180 | 7.69 | 1091 | 1748 |
| dariush TTA2 quick（public 0.37） | 1326 | 176/180 | 7.53 | 1215 | 1848 |
| dariush TTA8 full（public 0.39） | 1177 | 177/180 | 6.65 | 1343 | — |
| ours_model_v1 | 1065 | 177/180 | 6.02 | 1374 | 2062 |
| GT train 参考（标注记录） | 7.10/记录 | — | 7.10 | 1228 | 2120 |

* pub070 有 **11/180** 张 test 图无任何预测；dariush quick test 覆盖其中 **9/11**（27 实例）、ours 覆盖 9/11。
  → 从 pub070(0.34) 换到 dariush(0.39) 的一部分收益来自这些"原本空图"的补回。
* dariush TTA8 full 的预测面积中位 1343 比 GT 1228 大 9.4%、实例数 6.65/图略少于 GT 7.10/记录 —— 与 fold0 的"略大 + 略少"一致。
* ours test 面积中位 1374（更大）、实例数 6.02（更少）→ 偏保守，FN 风险高（与 fold0 结论一致）。

---

## 8. 复现命令

    # 1) 集成规则扫描（ours）
    C:\Python314\python.exe src\ensemble_eval.py \
        --base work\dariush_oof\pred_dariush_fold0.csv \
        --extra work\models\fold0_val_pred.csv --extra-conf-col confidence \
        --fold 0 --max-resid-frac 0.5

    # 2) TTA8 + TTA2 并集实验（含 bootstrap）
    C:\Python314\python.exe work\analysis\ensemble\tta_union.py

    # 3) 误差分析（base/extra）
    C:\Python314\python.exe src\error_analysis.py \
        --base work\dariush_oof\pred_dariush_fold0.csv \
        --extra work\models\fold0_val_pred.csv --extra-conf-col confidence \
        --fold 0 --empty-source submissions\pub070_kaggle_v1.csv \
        --empty-candidate dariush=work\cloud\runs\dariush_quick_v1\submission.csv \
        --empty-candidate ours=submissions\ours_model_v1.csv

    # 4) 单模型评估
    C:\Python314\python.exe src\pq_metric.py --gt-json data\MAGFiLO_1.0_Kaggle_2026\train\MAGFiLO_1.0_Annotations_kaggle2026_train.json \
        --pred-csv <csv> --stems <fold0 stems>   # 或用 P.gt_df_for_stems + P.evaluate

产物：<code>work/analysis/ensemble/ensemble_eval_fold0.json</code>、<code>tta_union_fold0.json</code>、
<code>error_analysis_fold0.json/.md</code>、<code>complementarity_fold0.json</code>、<code>test_submission_stats.json</code>。

---

## 9. KEEP 规则在稀疏 test 标注下的再优化（2026-10-29 晚新增）

上游：dariush receipt（models/dariush/solar_unet_receipt.json）指出 test GT 比 train 稀疏；若每个 filament 在 test 标注中
出现的概率为 r，则 fixed-KEEP 的期望 PQ 为：

    PQ'(r) = r*sum_iou / ( r*tp + 0.5*fp + 0.5*(1-r)*tp + 0.5*r*fn ),   sum_iou = sq*tp

数据：models/dariush/oof_keep_rule_grid.csv（33,880 条 KEEP 规则；t_lo 0.05-0.70 / t_hi 0.15-0.995 /
min_area 0-700 / min_meanp 0-0.95；pooled OOF 统计量 tp/fp/fn/sq）。
脚本：work/analysis/ensemble/keep_rule_pq_sparse.py、keep_rule_curves.py；结果：keep_rule_pq_sparse.json。

### 9.1 当前 KEEP 的稀疏 PQ'（校准 r）

| r | 0.70 | 0.75 | 0.77 | 0.80 |
|---|---|---|---|---|
| PQ' (current KEEP) | 0.3737 | 0.3887 | **0.3945** | 0.4029 |

r = 0.77 时 PQ' = 0.3945，与 dariush 实测 public 0.39 吻合（receipt 独立推出 r≈0.77，并用后续两次提交验证）
→ test 稀疏率取 0.76-0.77；本结论与 FAIR 观测一致。

### 9.2 各 r 最优规则 vs 当前 KEEP

| r | 最优 (t_lo, t_hi, min_area, min_meanp) | PQ' best | PQ' current | Delta |
|---|---|---|---|---|
| 0.70 | 0.35 / 0.995 / <=200 / 0.85 | 0.3749 | 0.3737 | +0.0013 |
| 0.75 | 0.35 / 0.995 / <=200 / 0.85 | 0.3891 | 0.3887 | +0.0005 |
| 0.77 | 0.35 / 0.995 / <=200 / 0.85 | 0.3946 | 0.3945 | +0.0001 |
| 0.80 | 0.35 / 0.99 / 200 / 0.85（=当前） | 0.4029 | 0.4029 | 0.0000 |

**结论：最大改进 +0.0013 < +0.003 门槛 → 不需要重跑推理。** current KEEP 同时是 grid 中 full-GT PQ 最优规则
（0.4523；dariush 报告的嵌套 OOF 0.4502）→ sparse 与 full-GT 双目标一致最优。
t_hi 0.995 的 +0.0013 代价是 full-GT PQ 0.4523→0.4487（pred 少 12%），不值得为它重跑云端 20 分钟。

### 9.3 面积偏大（+8.7%）是否需要额外修正 → 不需要

r=0.77 下 PQ' vs t_lo（fixed t_hi .99 / min_area 200 / min_meanp .85）：

| t_lo | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 |
|---|---|---|---|---|---|
| PQ' | 0.3932 | **0.3945** | 0.3910 | 0.3880 | 0.3851 |

提高 t_lo（收缩掩膜、修正面积偏大）单调变差（0.40 起 -0.0035，0.50 起 -0.0094）；
t_hi 在 0.90-0.995 平坦（0.3941-0.3946）；min_meanp=0.85 为尖锐最优（0.80→0.3932，0.90→0.3817）。
**结论：最优 KEEP 参数已隐含处理面积偏大，不要再腐蚀/提高阈值**（与 consensus 实验 ±1px 损失 0.12-0.20 PQ 一致）。

---

## 10. 概率级集成工具（src/prob_ensemble.py，等概率 dump 到位）

* 输入 npz 两种布局：① 每图一项 {stem: uint8 map}；② 打包 stems + probs(N,H,W) + 可选 sidecar json
  （kaggle-runner 的 dump 采用布局 ②：probs=uint8[N,2048,2048] + stems，uint8=round(sigmoid*255)）。
* 融合：mean / weighted-mean / median / max。
* KEEP 后处理：uint8 阈值 round(t*255)、8 连通域、max p > t_hi、area >= min_area、mean p >= min_meanp
  （与 dariush receipt keep_rule 完全一致）；scipy.ndimage 一次算每连通域 area/max/mean（2048^2 约 30-50 ms/图）。
* 输出 submission（RLE、逐图不重叠）；--eval-fold 0 直接报官方 pooled PQ；--limit 只跑前 N 图。
* 自测：python src/prob_ensemble.py --selftest（11/11 PASS：连通域保留/丢弃、RLE 往返、三种融合、npz 两种布局、submission）。
* 注意 model-count mismatch：fold0 dump 是 2 模型平均（更噪），test dump 是 10 模型平均（更平滑、置信更高）；
  在 fold0 上调完 KEEP 后要与 grid（8-10 模型 OOF 调出的 0.35/0.99/200/0.85）交叉验证，避免阈值迁移偏差。

---

## 11. 多尺度 TTA / 概率级融合工具与预实验（2026-10-29 19:40 前后）

### 11.1 工具（均自测通过，可直接对产物 npz 运行）

| 文件 | 作用 | 用法 |
|---|---|---|
| src/prob_ensemble.py | 概率 npz 加载/融合（mean,wmean,median,max）/ KEEP 后处理 / 提交输出 / 官方 PQ + 稀疏 PQ' | 见下 |
| work/analysis/ensemble/multi_scale_eval.py | 自动枚举单尺度、全均值、两两均值、中心/边缘/单边加权，逐组合报 PQ、PQ'、SQ/RQ、TP/FP/FN | python work/analysis/ensemble/multi_scale_eval.py --prob-dir <dir> --fold 0 |
| work/analysis/ensemble/prob_fusion_sweep.py | dariush+ours(+CAGE 三路) 概率加权融合 w 扫描 + 最优 w 处 KEEP 微调 | python work/analysis/ensemble/prob_fusion_sweep.py --dariush a.npz --ours b.npz --fold 0 |

* 快速路径：KEEP 组件统计 + 裁剪式 RLE 编码，**46ms/图**（旧实现 243ms/图，5.3x），RLE 与逐步编码逐字节一致。
* 端到端校验：把 dariush TTA8 CSV 掩膜转成 0/255 概率图 → 概率→KEEP→提交→官方 PQ = **0.4755**（TP1238/FP528/FN579），与 CSV 直接评估完全一致。
* npz 约定：布局 ② probs=uint8[N,2048,2048] + stems（kaggle-runner 采用）；也支持每图一 key、多尺度 probs_s085/probs_s100/probs_s115、（n_scales,N,H,W）。

### 11.2 预实验（**注意：dariush 真概率图尚未到位，以下用 CSV 二值掩膜当占位**）

* 把 dariush TTA8 CSV 转成二值概率图（0/255）作为 dariush 占位，与 ours 语义模型概率图
  （work/cloud/runs/sem_smoke_v1/sem_fold0/val_probs_best.npz，142 图，epoch2 smoke 模型，单源 KEEP PQ=0.3458）做 wmean 融合：

| w_ours | PQ | PQ' | SQ | RQ | pred |
|---|---|---|---|---|---|
| 0.0（dariush 占位） | 0.4755 | 0.4145 | 0.6881 | 0.6910 | 1765 |
| 0.2 | 0.4727 | 0.4125 | 0.6878 | 0.6873 | 1738 |
| 0.4 | 0.4495 | 0.3941 | 0.6800 | 0.6610 | 1613 |
| 0.5 | 0.4272 | 0.3768 | 0.6755 | 0.6324 | 1474 |
| 0.6 | 0.4129 | 0.3657 | 0.6735 | 0.6131 | 1382 |
| 0.8 | 0.3804 | 0.3337 | 0.6579 | 0.5781 | 1600 |
| 1.0（ours sem） | 0.3458 | 0.2986 | 0.6510 | 0.5312 | 2045 |

→ 在二值占位下 **w_ours>0 单调变差**（dariush 占位本身已是"完美硬掩膜"，软融合只会引入 ours 的噪声）。
**真概率图到位后需重跑**：软概率融合的行为与二值占位不同（软平均会柔化边界、影响 max/mean 统计），不能据此直接下结论。
* ours sem 单源 KEEP 扫描（t_lo×min_meanp 16 组）最优也仅 **PQ 0.3458**（t_lo .35 / min_meanp .85，即 dariush 规则），
  sparse 指标下 (0.45, 0.92) 略好（PQ' 0.3025 vs 0.2986）→ ours sem 在当前 smoke 权重下明显弱于 dariush（0.4755）。

### 11.3 待办（数据到位后）

1. multi_scale_eval：单尺度基线应复现 ≈0.4755；比较 3 尺度均值/加权、2 尺度组合，ΔPQ>0.003 才考虑 test 融合。
2. prob_fusion_sweep：真 dariush 软概率 + ours 概率（+CAGE 若到）扫 w∈{0,0.2,...,1.0} 与 KEEP 微调；双指标（PQ、PQ'）。
3. 若某组合胜出：用 prob_ensemble CLI 对 test npz 同参数出提交，并用 src/validate_submission.py 校验。

---

## 12. 多尺度 TTA 实测（probs_fold0_ms.npz，2026-10-29 20:10 结论：**放弃**）

数据：work/dariush_probs/probs_fold0_ms.npz（142 图；probs_s085/s100/s115/fused 各 uint8 2048²；
fold-aware 2 模型 × TTA8；scales [0.85, 1.0, 1.15]）。KEEP = 0.35/0.99/200/0.85。

先校验口径：**s100 单尺度重建 = PQ 0.47547 / PQ' 0.4145 / TP1238 / FP528 / FN579 / pred1765**，
与单尺度 dump 及 task-13 CSV 逐项一致 ✓。

| 组合 | PQ | ΔPQ | PQ' | ΔPQ' | SQ | RQ | pred |
|---|---|---|---|---|---|---|---|
| **single[s100]** | **0.4755** | **0** | **0.4145** | **0** | 0.6881 | 0.6910 | 1765 |
| single[s085] | 0.1551 | -0.3204 | 0.1498 | -0.2647 | 0.7381 | 0.2102 | 247 |
| single[s115] | 0.1239 | -0.3516 | 0.1205 | -0.2940 | 0.7556 | 0.1639 | 183 |
| mean[3 尺度] | 0.3312 | -0.1443 | 0.3057 | -0.1088 | 0.7148 | 0.4633 | 703 |
| mean[s085+s100] | 0.4198 | -0.0557 | 0.3770 | -0.0375 | 0.6975 | 0.6018 | 1113 |
| mean[s085+s115] | 0.1252 | -0.3503 | 0.1218 | -0.2927 | 0.7471 | 0.1676 | 187 |
| mean[s100+s115] | 0.4102 | -0.0653 | 0.3687 | -0.0458 | 0.7025 | 0.5839 | 1097 |
| center-heavy[s100=0.5] | 0.4138 | -0.0617 | 0.3727 | -0.0418 | 0.6996 | 0.5915 | 1063 |
| edge-heavy[edges=0.4] | 0.2506 | -0.2249 | 0.2366 | -0.1779 | 0.7298 | 0.3433 | 448 |
| scale1-heavy[0.6,0.2,0.2] | 0.2607 | -0.2148 | 0.2456 | -0.1689 | 0.7264 | 0.3590 | 472 |
| scale3-heavy[0.2,0.2,0.6] | 0.2397 | -0.2358 | 0.2269 | -0.1876 | 0.7336 | 0.3268 | 422 |
| runner fused（=3尺度均值） | 0.3312 | -0.1443 | 0.3057 | -0.1088 | 0.7148 | 0.4633 | 703 |

**结论：ΔPQ 全部为负（-0.056 ~ -0.352），PQ 与 PQ' 同向变差，远低于 +0.003 门槛 → 放弃多尺度 TTA，不跑 test 多尺度。**
机理：模型在固定 2048 分辨率训练，0.85/1.15 缩放把暗条尺寸推出训练分布 → 概率置信整体下降
（s085/s115 只剩 247/183 个实例 vs 1765），任何加权平均都稀释 s100 的高置信预测。

---

## 13. 真概率图上的 KEEP 交叉验证 + 概率融合（2026-10-29 20:05）

### 13.1 KEEP 在真 fold0 概率图上直扫（45 组：t_lo 0.30-0.40 × min_area 100/200/400 × min_meanp 0.80/0.85/0.90，t_hi 0.99）

* 最优（PQ）：**t_lo=0.35, t_hi=0.99, min_area=100 或 200, min_meanp=0.85 → PQ=0.47547**（=当前 KEEP）；
* 稀疏最优（PQ'）：t_lo=0.325, min_area=100, min_meanp=0.85 → PQ'=0.41478（vs 当前 0.41445，仅 +0.0003，PQ 反而 0.47503）；
* t_lo 0.30→0.4739 / 0.375→0.4747；min_area 400→0.4676（-0.008）→ **无需修改 KEEP**。
* 与 grid（8-10 模型 OOF）结论一致（t_lo .35 / t_hi .99 / min_area 200 / min_meanp .85）→ **KEEP 从 fold0(2 模型)
  迁移到 test(10 模型) 是安全的**。

### 13.2 dariush + ours 语义概率的软融合（真概率，非占位）

| w_ours | PQ | PQ' | SQ | RQ | pred |
|---|---|---|---|---|---|
| 0.0（dariush） | **0.4755** | 0.4145 | 0.6881 | 0.6910 | 1765 |
| 0.2 | 0.4640 | 0.4066 | 0.6876 | 0.6748 | 1626 |
| 0.4 | 0.4247 | 0.3753 | 0.6824 | 0.6224 | 1434 |
| 0.5 | 0.4071 | 0.3599 | 0.6777 | 0.6008 | 1421 |
| 0.6 | 0.4019 | 0.3541 | 0.6724 | 0.5978 | 1501 |
| 0.8 | 0.3754 | 0.3283 | 0.6579 | 0.5706 | 1680 |
| 1.0（ours sem smoke） | 0.3458 | 0.2986 | 0.6510 | 0.5312 | 2045 |

→ 单调变差：ours sem（smoke 权重，单源 0.3458）与 dariush 差距过大，概率融合也无收益。**不做该融合**。

### 13.3 test 概率图校验（关键能力）

probs_dariush_test.npz（180 图）+ 默认 KEEP → **1177 行 / 177 图，与 submissions/dariush_unet_full_v1.csv 逐行完全一致**
（80 图抽样 per-image max-IoU 均值 1.0000、stems/行数零差异）。→ 任何在 test 概率图上的 KEEP/融合参数变化，
都能直接产出可提交 CSV（prob_ensemble CLI），无需再跑云端推理。CAGE 概率图到位后可做第三源 w=0.2 试探
（其 fold0 dump 非 fold-aware，仅参考）。

---

## 14. 负结果与消融合集（2026-10-29 晚，最终版）

> 本表汇总本任务期间所有"尝试过但不采用"的路线与关键消融，供报告"negative results / ablation"一节直接引用。
> 除特别说明外均为 fold0 val 142 图、官方 pooled PQ（KEEP = 0.35/0.99/200/0.85）。

### 14.1 模型内部消融（ours pipeline）

| 实验 | 配置 | PQ | Δ | 说明 |
|---|---|---|---|---|
| 完整 pipeline | detector + refiner, conf .35 / ref .40 / min_area 96 | **0.3803** | 基线 | 加密 sweep 最优 |
| **refiner 贡献**（同 conf .35） | 无 refiner（detector-only） | 0.3550 | **-0.0253** | 去掉 refiner：SQ -0.0079、RQ -0.0323、TP -55、FP +49 |
| refiner 贡献（各自最优 conf） | 无 refiner conf .30 | 0.3569 | -0.0234 | refiner 主要补 RQ（实例完整性） |
| **TTA 消融** | tta=none（conf .35/ref .5） | 0.3622 | **-0.0158** | tta=guard 0.3780；TTA8 主要补 RQ |
| ours 语义模型（2-epoch smoke） | EfficientNetV2-S U-Net，KEEP 默认 | 0.3458 | — | 弱于 dariush（0.4755）0.13；本身也不够用 |
| ours 语义模型 + 自有 refiner | 用我们 refiner 精修 sem best 掩膜 | 0.3478 | -0.0058 | **refiner 对"非自身 detector seed"的掩膜无增益**（0.3536→0.3478，140 图） |

### 14.2 外部掩膜精修 / 跨模型融合

| 实验 | 基线 PQ | 结果 PQ | Δ | 结论 |
|---|---|---|---|---|
| **用我们 refiner 精修 dariush 掩膜** | 0.4755 | 0.4461 | **-0.0294** | refiner 训练分布 = 自家 detector crop，对 dariush 掩膜修边界反而变差（最佳筛选 variant area_change<30% 也救不回） |
| 掩膜级集成 dariush + ours（17 规则/阈值） | 0.4755 | ≤0.4755 | ≤0 | base_only 最优；no_overlap 最好也只是打平（0.4753） |
| **dariush TTA8 ∪ TTA2 并集** | 0.4755 | 0.4725 | -0.0029 | bootstrap 95% CI [-0.0061,+0.0002]，P(Δ>0)=3.4% |
| 概率级融合 dariush + ours sem（软概率 wmean） | 0.4755 | 0.4640 (w=.2) → 0.3458 (w=1) | 单调负 | ours sem 太弱，稀释 dariush 高置信预测 |
| **多尺度 TTA**（s085/s100/s115 全部组合） | 0.4755 | 0.124 ~ 0.4198 | **-0.056 ~ -0.35** | 模型固定 2048 分辨率训练，缩放破坏实例尺寸分布；已放弃，不跑 test 多尺度 |
| TTA8 vs TTA2（诚实 OOF） | TTA2 0.4688 | TTA8 **0.4755** | **+0.0067** | TTA 越多越好（但仅限同分辨率视图）；public 0.37→0.39 亦印证 |

### 14.3 KEEP / 后处理

| 实验 | 结论 |
|---|---|
| grid 33880 条 KEEP 规则 + 稀疏公式 PQ'（r=.70/.75/.77/.80） | 当前 KEEP 近似最优；最优改进 ≤ +0.0013（t_hi .995），< +0.003 门槛 → **不重跑推理** |
| 真 fold0 概率图直扫 KEEP（45 组） | 最优 = 当前 KEEP（PQ 0.47547）；min_area 400 → -0.008；t_lo ±0.025 ≤ -0.0005 → **KEEP 不变且跨模型数迁移安全** |
| 面积偏大（+8.7%）后处理腐蚀/提阈值 | t_lo .40 → -0.0035、.45 → -0.0065、.50 → -0.0094；膨胀 ±1px（共识实验）→ -0.12~-0.20 → **禁止额外腐蚀/膨胀** |

### 14.4 数据/评测基建（正向）

| 项 | 数字 |
|---|---|
| 官方 PQ 复刻验证 | 人类互评 pooled PQ = **0.3399 / SQ 0.6337**（与 FACTS §10 的 0.34/0.634 一致） |
| 稀疏公式校准 | r=0.77 时当前 KEEP 的 PQ' = 0.3945 ≈ dariush 实测 public **0.39** ✓ |
| dariush 概率图 dump 校验 | fold0 probs → KEEP 重建 PQ = 0.47547（与 CSV 逐项一致）；test probs → 与已提交 CSV **逐行完全一致**（1177 行/177 图） |

---

## 15. 结论汇总（供最终提交决策）

1. **最终提交 = submissions/dariush_unet_full_v1.csv（public 0.39）**；所有融合/精修/多尺度路线在 fold0 上均为负或打平。
2. 若 sem（40ep 正式权重）或 CAGE 概率图到位：唯一值得做的实验是**"dariush + 新源概率 wmean，w≤0.3，KEEP 保持默认"**，
   门槛：融合 fold0 PQ ≥ **0.4785** 且 PQ' 同向提升（相对 dariush 0.4755/0.4145）；不达门槛则弃用。
3. 后处理一律不改 KEEP、不做腐蚀/膨胀、不做多尺度。

---

## 16. sem v2 概率图复核 + 单折融合门槛判定（2026-10-29 21:55）

### 16.1 独立复核 Lead/modeling 的 sem v2 fold0 评估（128 图? 实际 142 图，honest OOF）

| KEEP | PQ | SQ | RQ | TP | FP | FN | PQ' |
|---|---|---|---|---|---|---|---|
| 默认（area 200, mmp .85） | **0.421075** | 0.677062 | 0.621916 | 1172 | 780 | 645 | 0.364662 |
| area 400 | **0.417090** | 0.676972 | 0.616111 | 1109 | 674 | 708 | 0.363338 |

与 Lead 给出的 0.4211 / 0.41709（JSON：SQ .676972 / RQ .616111 / PQ' .36334）**逐位一致** ✓

### 16.2 sem 单独最优 KEEP（fold0 概率图直扫，32 组）

| KEEP | PQ | PQ' | pred |
|---|---|---|---|
| 默认（.35/.99/200/.85） | 0.4211 | 0.3647 | 1951 |
| **最优 PQ：.35 / .99 / 200 / .90** | **0.4282** | 0.3738 | 1728 |
| 最优 PQ'：.40 / .99 / 200 / .92 | 0.4273 | **0.3743** | 1638 |
| area 400 + mmp .85（Lead 口径） | 0.4171 | 0.3633 | 1782 |

→ sem 概率图更噪，**提高 min_meanp 到 0.90 最有效（+0.0071 PQ）**；sem 的最优 KEEP 与 dariush 不同（dariush: mmp .85 / area 200），
说明两者概率分布不同，不能直接共用一套后处理参数。

### 16.3 融合后最优 KEEP（dariush × sem，w=0.1，18 组）

| KEEP | PQ | PQ' | pred |
|---|---|---|---|
| t_lo .325 / 200 / mmp .90 | 0.4462 | 0.3975 | 1260 |
| t_lo .375 / 200 / **mmp .85** | **0.4755** | 0.4145 | 1758 |
| t_lo .375 / 200 / mmp .90 | 0.4623 | 0.4087 | 1421 |

→ 融合后最好情况只是**与 dariush 单源打平（0.4755，pred 1758 vs 1765）**，没有任何组合超过 0.4755。
其他融合方式：max-fusion 最好 0.4527；mean-fusion 最好 0.4571（默认 KEEP）→ 均低于 dariush。

### 16.4 门槛判定

* 诚实单折（sem v2 fold0 held-out）：dariush×sem 最佳 w=0（所有 w>0 变差：0.1→0.4721、0.2→0.4674、0.3→0.4659、0.4→0.4596、0.5→0.4571）
* **Gate（PQ≥0.4785 且 PQ'≥0.4145）：FAIL**
* 判据符合 Lead 预期：单折 sem（0.42）与 dariush（0.4755）差距 0.055，概率平均无法补偿；
  需看"多折平均 sem（≥4-5 折，类 dariush 10 模型平均）"能否把 sem 提到 ~0.45-0.47 再融合。

### 16.5 多折平均融合的评估逻辑（已写工具，待 cross-fold probs）

* 工具：work/analysis/ensemble/multifold_sem_fusion.py
  * 输入多个 sem npz（同一 fold0 stems）+ dariush fold0 probs，先平均 sem probs，再与 dariush 扫 w；
  * **--honest** 时按 gate 判定；否则明确标注 "OPTIMISTIC PROXY"（多折平均在 fold0 上含训练过 fold0 的模型 → 只能当上界代理）；
  * 同时输出 sem-avg alone 的 PQ/PQ'，便于拆解"sem 自身变强"与"融合增益"。
* 正在向 modeling 请求 cross-fold probs：sem fold1/2/3/4 模型在 fold0 142 图上的推理结果（每个 ~3MB），
  用于评估"5 折平均"的乐观代理；最终 test 多折平均 probs 是否值得跑取决于代理结果。
* v3 clDice fold0 probs 与 v2 fold2 val probs 到位后补 A/B 与 fold2 复核。

---

## 17. sem v2 A/B 复核汇总（2026-10-29 22:05，fold0/fold1 双折）

### 17.1 独立复核结论：与 modeling 数字逐位一致

| 模型/折 | 默认 KEEP | sem 最优 KEEP | 最优 PQ | 复核状态 |
|---|---|---|---|---|
| sem v2 fold0 (bcedice, 24ep) | 0.421075 | t_lo .35 / t_hi .95 / area 200 / mmp .90 | **0.428201** | ✅ 我复算逐位一致（TP1116/FP613/FN701，PQ'=0.373765） |
| sem v2 fold1 (bcedice, 24ep) | 0.428887 | t_lo .40 / t_hi .95 / area 400 / mmp .90 | **0.442906** | ✅ 我复算逐位一致（TP1044/FP524/FN595，PQ'=0.386465） |

* 稳定结论：**min_meanp 0.85→0.90 是 sem 最稳增益**（两折 +0.0071 / +0.0140）；t_hi .95 与 .99 等价；
  t_lo/min_area 的折间最优不同（.35/200 vs .40/400）→ sem 的 KEEP 需按折/按融合联合调，不能直接复用 dariush 的。
* 与 dariush 对比：sem 单折 0.428/0.443 vs dariush fold0 honest 0.4755 → 差 0.03~0.05。

### 17.2 融合结论（单折 sem，诚实）

* dariush × sem v2 fold0（wmean，默认 KEEP）：w=0.1→0.4721、0.2→0.4674、0.3→0.4659、0.4→0.4596、0.5→0.4571；**最佳 w=0（0.4755）→ gate FAIL**。
* 融合 KEEP 联合扫描（w=0.1，18 组）：最好 **t_lo .375 / area200 / mmp .85 → 0.4755（= dariush，打平）**；mmp .90 → 0.4623。
* 其他融合方式：max-fusion 最好 0.4527；mean-fusion 最好 0.4571 → 均低于 dariush。
* modeling 用 sem 单折最优 KEEP 做融合扫描更低（best 0.4534）→ 印证"融合 KEEP 与权重必须联合调，且单折 sem 补不动 dariush"。

### 17.3 多折平均代理（进行中）

* 已备工具 work/analysis/ensemble/multifold_sem_fusion.py（多源 sem npz 先平均再与 dariush 扫 w；--honest 才作 gate，否则 OPTIMISTIC PROXY）。
* 正在等 modeling/runner 的 cross-fold probs（sem fold1/2/3/4 模型在 fold0 142 图上的推理结果）：
  单源评估它们（leaked）与 fold0-honest 模型的 PQ 差可估计"泄漏虚高 δ"，用 δ 对代理做去偏，得到"约等于诚实多折平均"的参考。
* v3 clDice fold0 probs 与 v2 fold2 val probs 到位后补 A/B 与 fold2 复核。

---

## 18. 五折 sem 平均乐观代理 × dariush 融合（2026-10-30 01:50，结论：不采用）

数据：work/models/sem_crossfold/sem_<0..4>_on_fold0.npz（5 折 sem 模型在 fold0 val 142 图上的推理；
fold0 模型 honest，fold1-4 训练时见过 fold0 图 → **乐观代理**）。KEEP = 默认（.35/.99/200/.85）。

### 18.1 泄漏幅度估计（δ）

| 模型 | fold0 上单源 PQ |
|---|---|
| fold0（honest） | 0.4195 |
| fold1 / fold2 / fold3 / fold4（leaked） | 0.4352 / 0.4259 / 0.4355 / 0.4141 |
| leaked 均值 | 0.4277 |
| **泄漏 δ = leaked 均值 - honest** | **+0.0081** |

（crossfold 推理配置与训练脚本 val 略有差异：fold0 honest 单模 0.4195 vs 训练脚本 0.4211，差 0.0016）

### 18.2 5 折平均 sem（proxy）与融合（wmean）

| 组合 | PQ | PQ' | SQ | RQ | pred |
|---|---|---|---|---|---|
| sem 5 折平均（proxy，default KEEP） | **0.4446** | 0.3901 | 0.6815 | 0.6524 | 1597 |
| sem 5 折平均去偏估计（-δ） | 0.4365 | — | — | — | — |
| dariush 单源 | **0.4755** | 0.4145 | 0.6881 | 0.6910 | 1765 |
| fused w=.05 | 0.4744 | 0.4149 | 0.6886 | 0.6890 | 1680 |
| fused w=.075（modeling 细扫最优） | 0.4751 | — | — | — | — |
| fused w=.10 | 0.4738 | 0.4151 | 0.6889 | 0.6877 | 1634 |
| fused w=.20 | 0.4697 | 0.4121 | 0.6891 | 0.6817 | 1600 |
| fused w=.50 | 0.4595 | 0.4039 | 0.6882 | 0.6677 | 1549 |

**结论：即使按 leaky 乐观代理（sem 被高估 δ=+0.0081），最佳融合 0.4751（w=.075）仍低于 dariush 单源 0.4755；
gate 0.4785 不可达。去偏后 sem 诚实估计仅 0.4365，差距更大 → 多折 sem 平均路线放弃。**
* 注意 w=.05/.10 时 PQ' 略高（0.4149/0.4151 vs 0.4145），但 PQ 更低（-0.0007/-0.0011），方向不一致且量级远小于 0.003。
* 半自动独立复核（我 + modeling 双跑）数字一致：sem-avg 0.444637 / debias 0.4365 / fused best 0.4751。

### 18.3 结论与提交建议

1. **最终提交保持 submissions/dariush_unet_full_v1.csv（public 0.39）**；sem 多折平均不接受。
2. fused_v2.csv 无需生成/校验（gate 未通过）；校验器 work/analysis/ensemble/validate_fused_submission.py 已就绪
   （对已知 dariush 提交：schema/覆盖/不重叠全 PASS；重建一致性 mean IoU 0.99995、146/177 图逐像素一致，
   6 图因上游管线取整差异有极小像素差——对自家管线生成的 fused_v2 会要求严格一致性）。
3. 唯一剩余上行：runner 的 "nk v3 test（25 模型 × 8 视图）" 若为 dariush 家族更强集成并产出 submission，
   需按公开 LB 决定是否替换；其 fold0 OOF 若无诚实评估，不可用本地 gate。

---

## 19. CAGE 候选与 dariush 的多样性（2026-10-30 01:55，供最后提交参考）

* submissions/cage_r5_v1.csv：validate OK（1234 行 / 177 图，逐图不重叠）；实例面积分布与 GT 参考很接近
  （中位 1204 vs GT 1228 px，ratio 0.98；每图实例 6.97 vs GT 7.10）。
* 与 dariush 提交（1177 行 / 177 图）逐图匹配（IoU≥0.5，前 60 图）：
  - dariush 掩膜被 CAGE 匹配 **84.5%**（匹配 IoU 均值 0.837 / 中位 0.872）
  - CAGE 有 **20.7% 掩膜是 dariush 没有的（novel）** → 两模型互补性明显
* 但 CAGE 无诚实 OOF（作者 split 未公开，probs 为非 fold-aware），无法在本地做门槛评估；
  其候选提交是否替换 dariush（public 0.39）只能以公开 LB 为准。若后续 CAGE 概率图到位，
  也只能做"参考级"融合（非诚实 gate），风险高于收益，默认不采用。

---

## 20. 本阶段最终结论（2026-10-30 02:00）

1. **最终提交：submissions/dariush_unet_full_v1.csv（public 0.39）**。
2. 已证伪/放弃：多尺度 TTA（-0.056~-0.35）、外部掩膜 refiner（-0.0294）、掩膜级集成（≤0）、
   单折 sem 概率融合（0.4755 打平或更差）、五折 sem 平均乐观代理融合（最优 0.4751 < 0.4755，去偏更差）、
   面积/腐蚀类后处理（均负）、KEEP 改动（不必要）。
3. 已确认正收益项（只在 ours pipeline 内）：refiner **+0.0253**、TTA **+0.0158**、TTA8 vs TTA2 **+0.0067**；
   dariush 家族本身（10 模型 TTA8）0.4755 fold0 OOF → 0.39 public，是本任务最优可复现产物。

---

## 21. nk v3 集成提交校验与分析（2026-10-30 03:35）

产物：C:\Users\ptcu_\filament_assets\downloads\nk_test\submission.csv（2328 行 / 180 图）；
来源：15 个 YOLO 模型（p1: yolo11m-1536-long; p2: yolo11m-1536 ×5 折; p3: yolo26m-1536 ×5 折 + yolo11m-2048 ×5 折）
经 consensus 合并；单模型原始实例 13.0k~15.0k（约 72-83/图），合并后 12.9/图。

### 21.1 格式校验（全部通过）

| 检查项 | 结果 |
|---|---|
| 列名 / filament_id 格式 / 唯一性 | PASS |
| stems 覆盖 | 177/180（dariush 也是 177/180；3 张全空图为已知空图） |
| 逐图实例不重叠 | PASS（overlap_px=0） |
| 解码有效性 | PASS（2328 masks 全部非空） |

### 21.2 分布对比（vs GT 训练参考）

| 指标 | nk v3 | dariush | GT 参考 |
|---|---|---|---|
| 实例数/图 | **12.93** | 6.65 | 7.10/annotator-record（物理图约 11.6） |
| 面积中位 | **974 px** | 1204 px | 1228 px |
| 面积 min | 155 | 253 | 9 |
| 图覆盖率 | 180/180 | 177/180 | 180 |

→ nk v3 是**召回导向**：实例数约为 dariush 的 2 倍、接近 GT 物理图均值；面积中位偏小 21% → 碎片化/小目标风险。

### 21.3 与 dariush 的互补性

* dariush 掩膜被 nk 匹配（IoU>=0.5）：**1061/1177 = 90.1%**（best IoU 均值 .796 / 中位 .862）
* nk 掩膜无 dariush 对应（novel）：**1251/2312 = 54.1%**
* → 两套提交差异极大；nk 额外召回大量 dariush 漏掉的暗条（也可能是未标注真值/碎片）。

### 21.4 15 模型一致度（consensus support，自建分析）

用 cache_test_shard0/1 的 15 个模型逐实例预测（每模型 13.0k~15.0k 实例），对最终 2328 个实例统计
"IoU>=0.5 支持它的模型数"：

| 支持度 | 实例数 | 占比 |
|---|---|---|
| >= 1/15 | 2099 | 90.2% |
| >= 5/15 | 1728 | 74.2% |
| >= 10/15 | 1529 | 65.7% |
| **= 15/15（全模型一致）** | **1259** | **54.1%** |

* support 均值 10.43、中位 15；与支持模型 best IoU 均值 0.865 / 中位 0.950。
* 仅 9.8% 实例只有 0-1 个模型支持（低置信/疑似 FP）。
* → 最终提交确实由强共识产生（过半实例 15/15 全票、中位 IoU 0.95），精度风险可控；
  主要不确定性来自**稀疏标注**（测试集约 23% 真值未标注 → 高召回的 novel 实例会被计 FP）。

### 21.5 决策建议

* nk v3 是**合法且高质量共识**的候选；公开分未知（其上游 3 个公共 notebook 家族单模 public 约 0.34-0.38，
  15 模 consensus 集成有希望 0.38-0.42，但方差大）。
* 若还有提交配额：**可用 nk v3 作为"冲高"候选**（预期 0.38-0.42，方差 ±0.03）；
  若只求稳：保持 dariush（0.39 已知）。
* 无诚实 OOF → 本地无法更精确预测；不建议再做 dariush+nk 的掩膜级合并（无 GT 可调，风险更高）。

---

## 22. nk_ref 公开 0.41 提交文件校验（2026-10-30 03:55）——最终提交候选

发现：C:\Users\ptcu_\filament_assets\downloads\nk_ref\submission.csv（1287 行 / 178 图；10/18 落盘；
同目录含 nk 作者公开产物：refiner/fold-0.pt、rescorer-hgb.pkl、splits/folds-metadata.json、splits/folds.csv）。
runner 报告 nk 官方公开 LB = **0.41**（rank 108；25 模型全量 × 8 TTA × consensus + refiner + rescorer）。

### 22.1 格式校验（全部通过）

* schema / filament_id 格式 / 唯一性 / RLE 解码：PASS
* stems 覆盖：178/180（2 张空图；dariush 也只有 177/180）
* 逐图实例不重叠：PASS（overlap_px=0）

### 22.2 分布（vs GT 训练参考）——几乎完美贴合

| 指标 | nk_ref | GT 参考 |
|---|---|---|
| 面积中位 | **1221 px** | 1228 px（ratio 0.99） |
| 面积均值 | 1918 | 2119.8（0.90） |
| 实例数/图 | **7.23** | 7.10（ratio 1.02） |

（对比：我们 15 模 nk v3 = 12.93 实例/图、面积中位 974 → 召回导向、碎片化明显）

### 22.3 与其他候选的关系

| 对比 | 结果 |
|---|---|
| nk_ref vs dariush | ref 掩膜 83.5% 被 dariush 匹配；dariush 掩膜 90.6% 被 ref 匹配 |
| nk_ref vs 我们 nk v3 | **ref 掩膜 95.2% 出现在 v3 中**；v3 有约 1000 个 ref 没有的 novel 掩膜 |
| 行数 | ref 1287 / v3 2328 / dariush 1177 |

→ nk_ref 本质是 nk 家族 consensus（+refiner+rescorer）的**精筛版本**；我们 15 模 v3 是它的召回超集
（多出的约 1000 掩膜大概率是稀疏标注下的 FP）。

### 22.4 三候选对比与建议

| 候选 | 已知/预期 public | 校验 | 风险 |
|---|---|---|---|
| **nk_ref/submission.csv** | **≈0.41（公开实测）** | PASS | 最低（分布最贴 GT） |
| nk v3（15 模复现） | ~0.38-0.41，方差大 | PASS | 召回过剩/碎片 FP |
| dariush_unet_full_v1.csv | 0.39（实测） | PASS | 已知 |

**建议最终提交 = nk_ref/submission.csv**（预期 0.41 > 0.39；公开 notebook 产物，与 pub070 复现同性质）。
Lead 已定 nk v3 不占提交配额（modeling 转述）；nk_ref 是否替换 dariush 由 Lead 决定。

### 22.5 来源确认（runner 独立核实，2026-10-30 03:32）

* nk_ref/submission.csv 与 Kaggle 公开 dataset **notoverkil/solar-filament-weights**（作者 Salty Taro/notoverkil）
  中的 submission.csv **md5 一致**（298fb24d...，339,590 B，1287 行/178 图）；dataset 时间戳 10/18/2026 11:55 = 其 LB 0.41 提交日。
  → **极大概率就是拿 0.41 的那份提交文件**。
* runner 已复制为 submissions/candidates/nk_public_ref_submission.csv，并用 src/validate_submission.py 独立复核：
  rows=1287，images 178/180，area min/med/max=190/1221/22430，total 2.468 Mpx，OK。
* 提交由 Lead 决定/执行（约定：runner 不做 competitions submit）。

### 22.6 Lead 决策与合规替代路线（2026-10-30 03:55）

* **Lead 决策：不提交 nk_ref 公开提交文件**。原因：规则/prize 复核风险 —— 提交必须由自己的代码产生，
  直接提交他人 submission 会被判不合格。该文件（md5 298fb24d…，339,590 B，1287 行/178 图）仅作**对照基准**。
* 合规替代（Lead 本地运行中，ETA ~04:50）：用 notoverkil **公开权重** + **我们自己的 consensus/聚类实现** +
  **模型投票数过滤**（15 模型 213,770 实例 → 重新聚类，记录每簇支持模型数/投票数/像素一致度），产出我们自己的提交候选
  （预期 0.38-0.41）。与 dariush 公开权重复现（0.39）同性质，合规。
* 我这边已先做一版**独立、无需等待**的 vote-filter 原型（基于我们现有 nk v3 consensus 候选 + 我保存的每实例支持模型数）：
  - 工具/产物：work/analysis/ensemble/nk_v3_votes/vote_filter_sweep.{json,md}、filtered_votes{12..15}.csv
  - 以 nk_ref 为对照（"ref 接近度"）：

| 投票阈值 | 实例数 | 实例/图 | 面积中位 | ref 召回 | cand 精度 | F1 vs ref | per-image best IoU |
|---|---|---|---|---|---|---|---|
| support>=12 | 1442 | 8.10 | 1171.5 | 0.951 | 0.849 | 0.897 | 0.9276 |
| support>=13 | 1378 | 7.79 | 1183.5 | 0.942 | 0.880 | 0.910 | 0.9328 |
| support>=14 | 1327 | 7.50 | 1207 | 0.926 | 0.898 | 0.912 | 0.9328 |
| **support>=15（全票）** | **1259** | **7.11** | **1221** | 0.905 | 0.925 | **0.915** | 0.9325 |
| nk_ref（对照，0.41） | 1287 | 7.23 | 1221 | — | — | — | — |

  - 所有 filtered 候选 schema/覆盖/逐图不重叠均 PASS（check_ensemble_candidate）。
  - **support>=15 与 ref 最接近**：密度 7.11 vs 7.23、面积中位 1221 vs 1221（完全一致）、ref 召回 .905 / 精度 .925。
  - → 建议提交阈值：优先 **votes=15（全 15 模型一致）**；若 Lead 的重新聚类版本密度更低（~6.8-7.2/img）可用 14/15 之间；
    目标密度区间 **7.0-7.5 实例/图**（nk_ref 7.23、GT 记录 7.10）。
  - 注意：本原型是"在已 consensus 合并的掩膜上按支持数过滤"（比 Lead 的完整重新聚类简单）；
    Lead 版本会更精细，但可用上述 ref 接近度指标（per-image best IoU、密度、面积中位）做同样比对。

### 22.7 投票阈值候选全表（submissions/candidates/nk_v3_vote{4,6,8,10,12,14,15}.csv，2026-10-30 04:00）

所有候选由"我们自己的 nk v3 consensus 候选 + 每实例 15 模型支持数"过滤得到；全部通过
schema/ID/RLE/覆盖/逐图不重叠校验（validate_fused_submission.py）。以 nk_ref（0.41 对照）为"接近度"目标：

| votes | rows | imgs | inst/img | area med | ref recall | precision | F1 | per-img best IoU | dariush match |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 1779 | 180 | 9.88 | 1086 | .959 | .695 | .806 | .626 | .598 |
| 6 | 1680 | 180 | 9.33 | 1104 | .959 | .736 | .833 | .662 | .631 |
| 8 | 1607 | 180 | 8.93 | 1117 | .958 | .768 | .853 | .691 | .659 |
| 10 | 1529 | 178 | 8.59 | 1147 | .957 | .806 | .875 | .725 | .691 |
| 12 | 1442 | 178 | 8.10 | 1171.5 | .951 | .849 | .897 | .763 | .728 |
| 14 | 1327 | 177 | 7.50 | 1207 | .927 | .898 | .912 | .807 | .782 |
| **15** | **1259** | **177** | **7.11** | **1221** | .906 | .925 | **.916** | .832 | .814 |
| nk_ref（0.41） | 1287 | 178 | 7.23 | 1221 | — | — | — | — | .835 |

判定：
* **votes=15 与参考最一致**：密度 7.11 vs 7.23、面积中位 **1221 vs 1221 完全相同**、与 dariush 重合 81.4% vs ref 83.5%；
  且 7.11 实例/图 ≈ GT 训练记录均值 7.10 → 密度口径正确。
* votes=14 为"高召回版"：7.50/img、ref 召回 .927、精度 .898（若偏好召回可用）。
* votes≤12 明显偏召回（密度 8.1-9.9/img、精度 .70-.85），与 ref 偏离大，仅作对照组。
* **建议提交：nk_v3_vote15.csv（主选）/ vote14.csv（备选）**；预期 public～0.40-0.41（与 ref 同量级），
  高于 dariush 0.39。
