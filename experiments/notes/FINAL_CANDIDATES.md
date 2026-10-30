# sem 5 折 × dariush 融合候选（2026-10-30 01:20）

输入：`work/cloud/runs/probs_dump_v1/probs_dariush_test.npz`（180 图 dariush TTA8 概率）
+ `C:\Users\ptcu_\filament_assets\probs\sem_test_probs_5fold.npz`（180 图，5 折 sem 概率平均，uint8 2048²）
+ default KEEP（t_lo .35 / t_hi .99 / min_area 200 / min_meanp .85），wmean 加权 [1, w]。

生成命令（可复现）：

    python src/make_sem_fusion_candidates.py \
        --dariush-probs work/cloud/runs/probs_dump_v1/probs_dariush_test.npz \
        --sem-probs C:/Users/ptcu_/filament_assets/probs/sem_test_probs_5fold.npz \
        --weights 0.05 0.075 0.10 0.15 0.20 0.25 0.30 0.40 0.50 \
        --keep 0.35 0.99 200 0.85 --out-dir submissions/candidates

## 统计（全部 validate=0）

| w | 文件 | rows | images | 保留 dariush(IoU>0.9) | 新增(IoU<0.5) | 面积中位 |
|---|---|---|---|---|---|---|
| 0.05 | fused_w005.csv | 1155 | 177 | 1152 | 1 | 1359 |
| 0.075 | fused_w008.csv | 1140 | 177 | 1134 | 2 | 1366 |
| 0.10 | fused_w010.csv | 1136 | 177 | 1126 | 5 | 1364 |
| 0.15 | fused_w015.csv | 1123 | 176 | 1112 | 5 | 1369 |
| 0.20 | fused_w020.csv | 1115 | 176 | 1101 | 6 | 1378 |
| 0.25 | fused_w025.csv | 1118 | 176 | 1097 | 12 | 1376 |
| 0.30 | fused_w030.csv | 1117 | 176 | 1094 | 13 | 1373 |
| 0.40 | fused_w040.csv | 1107 | 176 | 1083 | 14 | 1377 |
| 0.50 | fused_w050.csv | 1100 | 176 | 1070 | 16 | 1380 |
| sem-only | sem_only.csv | 1147 | 174 | 771 | 116 | 1318 |

## 依据（fold0 诚实校准）

- dariush fold0 OOF 0.475472；fold0 融合曲线（default KEEP）：w=0.05 → 0.474640；0.075 → 0.473190；
  0.10 → 0.473086；0.125 → 0.472567；0.15 → 0.469668；0.20 → 0.468036；0.30 → 0.467230（单折 sem 补不动 dariush）。
- 5 折平均预期比单折 +0.010~0.020（单折 KEEP 最优均值 0.4302）→ 预期最优 fused 0.475-0.481 @ w=0.05-0.10，gate 0.4785 边际。
- 掩膜级融合（sem masks + dariush masks 的 recover/consensus）全部低于基线，已否掉。

## 推荐提交顺序（配额 3）

1. `submissions/candidates/fused_w008.csv`（w=0.075）
2. `submissions/candidates/fused_w010.csv`（w=0.10）
3. `submissions/candidates/fused_w005.csv`（w=0.05，保底）
4. 备用：fused_w015.csv

sem_only 仅作诊断（显著低于基线）。