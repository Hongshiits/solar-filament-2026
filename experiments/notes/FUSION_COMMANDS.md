# 最终提交命令清单（22:57–23:10 整理，modeling）

所有大产物在 C:\Users\ptcu_\filament_assets\；小 CSV/报告留 D: 项目。

## A. sem 5 折 test probs（云端 kernel，T4x2）

### A1. 权重 dataset（本地打包目录）

    C:\Users\ptcu_\filament_assets\weights\sem_weights\
        dataset-metadata.json            (id: ptcuoigl/filament-sem-weights)
        sem_fold0_bcedice.pt             (85.2 MB)
        sem_fold1_bcedice.pt             (85.2 MB)
        sem_fold2_bcedice.pt             (85.2 MB)
        sem_fold3_bcedice.pt             <- 等 sem folds3/4 kernel（ETA ~00:25）到位后加入
        sem_fold4_bcedice.pt             <- 同上
        sem_fold0_bcedice_cldice.pt      (A/B 用，kernel 会自动跳过 *_cldice.pt)

    kaggle datasets create -p C:\Users\ptcu_\filament_assets\weights\sem_weights --dir-mode zip
    kaggle datasets version -p C:\Users\ptcu_\filament_assets\weights\sem_weights -m "add fold3/4" --dir-mode zip

### A2. kernel 目录（已写好）

    work/cloud/kernel_sem_test_probs/
        run_sem_test_probs.py    # 自包含。父进程把 180 张图按图片拆给 2 个 GPU 子进程；每个子进程都跑全部 5 个 fold 模型
                                 # (tile=1024 overlap=128 tta=2)，各自写 shard npz；父进程合并 → sem_test_probs.npz
        kernel-metadata.json     # id ptcuoigl/filament-sem-test-probs; competition_sources filament-segmentation-2026;
                                 # dataset_sources ptcuoigl/filament-cloud-code-v2 + ptcuoigl/filament-sem-weights;
                                 # machine_shape NvidiaTeslaT4x2（若 Kaggle 不识别则 enable_gpu=true 单卡，脚本 --shards 1 也可跑）
        README.md

    kaggle kernels push -p work/cloud/kernel_sem_test_probs
    kaggle kernels status ptcuoigl/filament-sem-test-probs
    kaggle kernels output ptcuoigl/filament-sem-test-probs -p work/cloud/runs/sem_test_probs_v1
    Copy-Item work/cloud/runs/sem_test_probs_v1/sem_test_probs.npz C:\Users\ptcu_\filament_assets\probs\sem_test_probs_5fold.npz

预期 30–45 min（2×T4）；输出 layout (a) {stem: uint8[2048,2048]}，180 图。

## B. 融合提交（本地 CPU，一键）

### B1. 先用诚实 fold0 校准（dariush fold0 probs × sem fold0 val probs）

    python src/fuse_probs_sweep.py --prob-npz work\cloud\runs\probs_dump_v1\probs_dariush_fold0.npz work\models\sem\sem_0_bcedice_val_probs_best.npz --extra-weights 0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5 --fuse wmean --keep 0.35 0.99 200 0.85 --fold 0 --out-dir work\models\fusion_calib
    # 两轮 KEEP 联合扫描：固定 w 扫 KEEP（t_lo .30/.35/.40, t_hi .95/.99, min_area 100/200/400, min_meanp .85/.90）→ 固定 KEEP 回扫 w

### B2. test 融合（w 网格 0.10–0.50，每个 w 一份候选 + 自动 validate）

    $dariush = "work\cloud\runs\probs_dump_v1\probs_dariush_test.npz"
    $sem = "C:\Users\ptcu_\filament_assets\probs\sem_test_probs_5fold.npz"
    foreach ($w in @(0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)) { $tag = [string]::Format("{0:0.00}", $w); python src\make_fused_submission.py --dariush-probs $dariush --sem-probs $sem --weights 1.0 $w --fuse wmean --keep 0.35 0.99 200 0.85 --out ("submissions\fused_sem_dariush_w" + $tag + ".csv") }

最终提交文件：submissions/fused_sem_dariush.csv（= 诚实 fold0 上最好的 w 那份改名）；每份 CSV 已由 src/validate_submission.py 校验。

### B3. 单份快速命令（最常用）

    python src/make_fused_submission.py --dariush-probs work\cloud\runs\probs_dump_v1\probs_dariush_test.npz --sem-probs C:\Users\ptcu_\filament_assets\probs\sem_test_probs_5fold.npz --weights 1.0 0.30 --fuse wmean --keep 0.35 0.99 200 0.85 --out submissions\fused_sem_dariush.csv

## C. nk ensemble 候选（等 fold0 结果）

### C1. fold0 诚实 PQ（云端 kernel 已跑，产物在 C:\Users\ptcu_\filament_assets\downloads\nk_fold0）

    期望文件：fold0_results.json（per-family + cross-family consensus PQ）、fold0_consensus_pred.csv。
    gate：cross-family consensus PQ ≥ 0.46 → 值得跑 test 阶段（Lead 规则）。

### C2. test 产物校验（submission schema/覆盖/逐图不重叠/面积分布）

    python src\check_ensemble_candidate.py --submission C:\Users\ptcu_\filament_assets\downloads\nk_test\submission.csv --reference submissions\dariush_unet_full_v1.csv --out work\models\nk_candidate_check.json

### C3. nk × dariush 融合对照（先 fold0 评估，再决定是否用于 test）

    # eval：dariush fold0 probs + nk fold0 consensus pred CSV；报 dariush/nk/nk_plus/recover/consensus 五种策略 PQ
    python src\fuse_ensemble_with_dariush.py --mode eval --dariush-probs work\cloud\runs\probs_dump_v1\probs_dariush_fold0.npz --ensemble-pred C:\Users\ptcu_\filament_assets\downloads\nk_fold0\fold0_consensus_pred.csv --out-dir work\models\ensemble_fusion

    # submit：dariush test probs + nk test submission.csv（选 eval 里最好的策略，默认 recover）
    python src\fuse_ensemble_with_dariush.py --mode submit --dariush-probs work\cloud\runs\probs_dump_v1\probs_dariush_test.npz --ensemble-pred C:\Users\ptcu_\filament_assets\downloads\nk_test\submission.csv --final-strategy recover --out submissions\fused_nk_dariush.csv

## D. 当前运行/等待

- sem v3 评估：fold0 clDice best official PQ 0.411483（默认 KEEP 即最优）→ bcedice 胜（0.4211 默认 / 0.4282 最优）；fold2 bcedice baseline 0.426439，KEEP 扫描中。
- nk ensemble fold0 kernel：~23:15 出（Lead 转达）。
- sem folds3/4 kernel：~00:25 出。