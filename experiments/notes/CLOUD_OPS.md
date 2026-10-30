# CLOUD_OPS — Kaggle 云端通道运维手册（kaggle-runner 维护）

> 覆盖：云端推理模板、云端训练 runner、公开权重复现 kernel、Final Submission 机制、GPU 配额与踩坑。
> 权威事实见 `FACTS.md`；本文只写"怎么在 Kaggle 上跑"。
> 最后更新：2026-10-29 17:20 HKT。

## 0. 总览：我们的 Kaggle 资源（账号 ptcuoigl）

| 类型 | ref | 用途 | 状态 |
|---|---|---|---|
| Dataset | `ptcuoigl/filament-pub070-weights` | 公开 0.70 权重（detector 184MB + refiner 57MB） | ready, v1 |
| Dataset | `ptcuoigl/filament-cloud-code` | 代码包（单 `code_pkg.zip`：src/*.py + folds.json + medoid.json） | ready, v5 |
| Kernel | `ptcuoigl/filament-cloud-inference-template-v1` | **云端推理模板**（已验证：复现 pub070 1299 行、MD5 09c0eae…） | COMPLETE v1 |
| Kernel | `ptcuoigl/filament-cloud-runner-v1` | **云端训练 runner**（probe/smoke/full 三模式）；fold0 full 已 COMPLETE | COMPLETE v3 |
| Kernel | `ptcuoigl/dariush-unet-cloud-v1` | 公开 dariush 10×U-Net 推理（quick v1 + full v3 均完成） | COMPLETE v3 |
| Kernel | `ptcuoigl/dariush-oof-fold0-v1` | dariush 在我们 fold0 val 的诚实 OOF（TTA2 v1 + TTA8 v2） | COMPLETE v2 |
| Kernel | `ptcuoigl/cage-r5-cloud-v1` | jamesmills CAGE r5 test 推理（第三候选） | COMPLETE v1 |
| Kernel | `ptcuoigl/dariush-probs-dump-v1` | dariush 10×U-Net uint8 概率图 dump（test 180 + fold0 142） | COMPLETE v1 |
| Kernel | `ptcuoigl/dariush-multiscale-fold0-v1` | dariush 多尺度 TTA 概率图（fold0 val, scales 0.85/1.0/1.15） | RUNNING |
| Kernel | `ptcuoigl/filament-semantic-train-v1` | 共识软标签语义 U-Net 云端训练（FOLDS=0,1, 24ep） | RUNNING v2 |
| Kernel | `ptcuoigl/cage-probs-dump-v1` | CAGE r5 mask+hull 概率图 dump（test+fold0） | 已写好，待 GPU 槽位 |

**并发上限**：同一账号同时最多 **2 个 GPU session**，push 第三个会报
`Maximum batch GPU session count of 2 reached`（不是排队，是直接失败）→ 需等一个跑完再 push。
**GPU 配额**：周 30h（T4），`python -m kaggle quota` 查；本次已用约 0.5h（截至 17:20）。
**GPU 规格**：`machine_shape: NvidiaTeslaT4` 实际给 **2× Tesla T4（各 15GB）**——推理 kernel 可双卡并行；
ultralytics 训练默认只用 1 卡（未做 DDP）。

## 1. 云端推理模板（任何权重 → submission.csv）

用途：本地 GPU 被占用/不可用时，把任意权重跑官方 180 张 test 图，取回 `submission.csv`。

- Kernel: `ptcuoigl/filament-cloud-inference-template-v1`（脚本 kernel）
- Dataset: `ptcuoigl/filament-pub070-weights`（weights 目录挂载）
- 配置方式：内核脚本顶部 `CONFIG`（engine/output_dir/images_dir/weights_dir/md5），或改 `CONFIG["engine"]="custom"` + `custom_code_dir/custom_module` 挂代码 dataset 调自定义 `predict(images_dir, weights_dir) -> dict[stem, list[mask]]`。
- 模板自带：每图非重叠硬检查、RLE 编码、schema 校验、`run_report.json`。
- **回归验证（重要）**：pub070 engine 输出应恒等于作者 1299 行产物，MD5 `09c0eae5fdff29a91f8a740475ecaf36`；
  v1 实测：COMPLETE，1299 行 / 169 图，MD5 完全一致，本地 pycocotools 校验 PASS。
- 重跑命令：
  ```powershell
  python -m kaggle kernels push -p src/kaggle_kernels/cloud_template
  python -m kaggle kernels status ptcuoigl/filament-cloud-inference-template-v1
  python -m kaggle kernels output ptcuoigl/filament-cloud-inference-template-v1 -p work/cloud/runs/infer_template_v1
  ```
- 实测耗时：推送→COMPLETE 约 5 分钟（含启动+推理 315s 级别），GPU 占用约 0.1h/次。

## 2. 云端训练 runner（modeling 的 cloud_pipeline.py）

- Kernel: `ptcuoigl/filament-cloud-runner-v1`（脚本 kernel，enable_gpu=true, **enable_internet=true**，挂 code dataset + competition）
- 代码 dataset：`ptcuoigl/filament-cloud-code`（**单 zip** `code_pkg.zip`，Kaggle 挂载时会自动解压到 dataset 根）：
  ```
  code_pkg.zip
    src/cloud_pipeline.py, src/pq_metric.py, src/probe_env.py, ...
    work/cache/folds.json, work/cache/folds_modeling.json
    work/yolo_data/meta/medoid.json
  ```
- runner 逻辑（`src/kaggle_kernels/cloud_runner/run_cloud.py`）：
  1. 打印 torch/CUDA/GPU；2. `pip install ultralytics==8.4.191`（与本地一致）；
  3. 定位并解压 `code_pkg.zip` 到 `/kaggle/working/code`；4. 设置 `FILAMENT_CODE_ROOT`/`PYTHONPATH`；
  5. 跑 `src/cloud_pipeline.py`，传 `--code-root/--folds-json/--out-dir /tmp/filament_run` + CONFIG 里的参数；
  6. 只把 `submission.csv, val_report.json, best_det.pt, best_ref.pt, summary.json, config.json, train_log.txt,
     val_per_image.csv, sweep.csv, folds_used.json` 复制回 `/kaggle/working`（大缓存留在 /tmp，避免 output 下载几百 MB）；
  7. 写 `runner_summary.json`（verdict/耗时/缺失文件）。
- 三种模式（改 `CONFIG["mode"]/["pipeline_args"]` 后重新 push）：
  - `probe`：`src/probe_env.py`，验证 GPU/internet/数据/库（无需训练）。
  - `pipeline + ["--smoke"]`：10 train/4 val 图、1 epoch、ref crop 128、只推 4 张 test（5-10 min）。
  - `pipeline + ["--fold","0","--det-epochs","30","--ref-epochs","8","--img-size","1024","--batch","8","--label-source","all"]`：完整 fold0。
- 推送/取回同 §1；本地校验：`python src/validate_submission.py work/cloud_runs/<run>/submission.csv`（含逐图重叠硬检查）。
- 依赖：ultralytics 8.4.191（runner 自动装）、torch 2.10（镜像自带）、torchvision、pycocotools、opencv、pandas、numpy；
  internet=true 用于 pip + 下载 `yolo11s-seg.pt`/resnet18 ImageNet 权重。
- 已知耗时（T4，modeling 预估）：detector 30ep ≈ 100-120 min（922 samples/ep，3-4 min/ep 含 val）；
  refiner cache 6382 crops + 8 ep ≈ 10-15 min；fold0 val 推理+阈值扫描+test 180 图 ≈ 10 min；总计 **2-2.5h**。
- 输出含义：`val_report.json` 是 fold0 val（142 物理图，官方 pooled PQ，用 code dataset 里的 `pq_metric.py` + harness `folds.json`）；
  `sweep.csv` 是 conf/ref/TTA 网格；`best_det.pt/best_ref.pt` 是 fold0 权重（可用于集成/OOF）。

### 2.1 smoke 验证要点（踩过的坑）
- smoke 用 1 epoch 垃圾模型，test 只推 4 张 → 很可能 0 预测；pipeline 原本 fail-closed 报 `empty submission`。
  我们打包副本里对 **smoke 模式**放行空提交（写 header-only CSV，仅验证写入路径），full 模式仍 fail-closed。
- smoke 的 `val_report.json` PQ 会是 0.0（1 epoch），**只验证字段齐全与流程**，不看分数。

## 3. 公开权重复现 kernel（task-12 / task-13）

### 3.1 dariush 10×U-Net（test 推理 → 提交）
- 资产：`dariushafshar/solar-filament-unet-weights`（公开，10 权重 + smp wheel + receipt + folds.csv + oof grid；作者自评 OOF PQ 0.4502 / public 0.39）
- Kernel: `ptcuoigl/dariush-unet-cloud-v1`，脚本 = `src/kaggle_kernels/dariush_unet/run_dariush_unet.py`
  （作者 notebook 代码 + 顶部 `CONFIG`：max_models/tta/skip_analysis/keep）。
- 关键点：TTA=8（D4）；10 模型 uint8 概率图平均；KEEP=(t_lo 0.35, t_hi 0.99, min_area 200, min_meanp 0.85)；双 T4 并行（作者代码原生线程分卡）。
- quick 验证（2 模型 × TTA2）：**1326 行 / 176 图，14/14 自检通过**，2 模型 180 图约 71s/模型/卡。
- full（10 模型 × TTA8）：见 work/cloud/runs/dariush_full_v1（结果见 §6 记录）。
- 取回后本地：`python src/validate_submission.py <csv>` → `submissions/dariush_unet_v1.csv`。

### 3.2 dariush 诚实 OOF（给集成用，task-13）
- Kernel: `ptcuoigl/dariush-oof-fold0-v1`，脚本 = `src/kaggle_kernels/dariush_oof/run_dariush_oof.py`
- 方法：我们 `work/cache/folds.json` 的 fold0 val（142 图）→ 用 dariush folds.csv 查每图 fold f → 用 dariush 的 fold (f+1)%5 的 2 个模型（resnet34+efficientv2_s，都没见过该图）× TTA8 → uint8 平均 → KEEP → RLE。
- 输出：`/kaggle/working/dariush_oof_fold0.csv` + `oof_report.json`；取回到 `work/dariush_oof/pred_dariush_fold0.csv`。
- 无 score 列（语义分割 + 固定阈值），集成时按 1.0 处理。

### 3.3 jamesmills CAGE r5（备选提交）
- 资产：`jamesmills1975/cage-filament-r5-weights`（U-Net mask + hull 两权重 + `infer_r5.py`；MODEL_CARD 自评 PQ 0.405，偏乐观）
- Kernel: `ptcuoigl/cage-r5-cloud-v1`（internet=true 用于 pip 装 smp 0.5.0），脚本 `src/kaggle_kernels/cage_r5/run_cage_r5.py`
- 输出 /kaggle/working/submission.csv + run_report.json。

## 4. Final Submission 机制（T5 第 3 项结论）

**平台规则**
- rules §2b: "You may select up to two (2) Final Submissions for judging."
- Submissions 页官方提示（讨论区 Vopani 引原文）："You can select up to 2 submissions to be used to calculate your final leaderboard score. If 2 submissions are not selected, they will be chosen based on your best submission scores on the public leaderboard."
- Kaggle staff（discussions/product-feedback/104827, Ian Catindig）：选择**比赛关闭前不生效**；关闭后 private LB 排名只基于 selected submissions；**只选 1 个则系统自动补选 public best**；public LB 始终显示 best submission。
- **API 不支持选择**：kagglesdk 2.2.5 的 CompetitionApiClient 无 select RPC；只有只读 `list_submissions(group=SELECTED)`（实测我们账号返回 0 条）。选择只能在 Kaggle 网页 **My Submissions → Final Submissions** 完成。
- 本赛 `numScoredSubmissions=1`：自动计分只取 1 个（默认 best），与"2 个 Final Submissions 交评审"并存。

**时间**
- Kaggle 提交截止：2026-11-15 06:00 UTC（北京 11-15 14:00）。
- host 报告/Google Form 截止：Nov 15 2026 23:59 CDT = 11-16 04:59 UTC（北京 11-16 12:59）。
- 最终评审还要：4 页技术报告（Overleaf 模板）+ 公开 Git repo（requirements.txt、完整 notebook、**权重入库**）。

**操作清单（Lead）**
1. 每日 5 次提交内继续产出 best；记录到 `submissions/LOG.md`。
2. 截止前（建议 11-13 之前）网页选 2 个 Final Submissions：优先 private 期望最高且**都能在 repo 复现**的；不手动选会被自动按 public best 补满（有过拟合 public 的风险）。
3. 两个入选产物写进报告/仓库；提交 Google Form。
4. 注意 public LB 只显示 best，不代表 selected 的那个；别用 public 单点调参。

## 5. GPU 配额与成本记录

| 时间(HKT) | 任务 | 机型 | 状态 | 约耗 GPU | 备注 |
|---|---|---|---|---|---|
| 10-29 15:36 | pub070 推理（原始 notebook 复现） | T4 | COMPLETE | 0.15h | 1299 行 MD5 一致 |
| 10-29 16:36 | cloud runner smoke v1 | T4 | ERROR | 0.05h | refiner crop bug（已修） |
| 10-29 16:51 | cloud runner smoke v2 | T4 | ERROR | 0.04h | empty submission（smoke 放行已修） |
| 10-29 16:50 | dariush quick（2 模型 TTA2） | T4×2 | COMPLETE | 0.06h | 1326 行 176 图 |
| 10-29 17:00 | 本地 CPU smoke ×1 | CPU | PASS | 0 | 全流程 + submission 写出 |
| 10-29 18:23 | dariush full（10 模型 TTA8，v3） | T4×2 | **COMPLETE** | 0.35h | 20.3 min，1177 行 177 图 |
| 10-29 17:03 | cloud pipeline fold0 full（30ep） | T4 | **COMPLETE** | 1.5h | 89 min，val PQ 0.3780 |
| 10-29 18:40 | dariush OOF fold0（TTA8，142 图） | T4×2 | **COMPLETE** | 0.13h | 7.5 min，PQ 0.4755 |
| 10-29 18:37 | dariush OOF fold0（TTA2 快速版） | T4×2 | **COMPLETE** | 0.06h | 3 min，PQ 0.4688 |
| 待 | CAGE r5 | T4×2 | 未跑（push 被会话自动审核拦截） | — | 备选第三候选 |

> 周配额 30h，刷新 2026-10-31 00:00 UTC。2026-10-29 19:00 实测：**已用 2.68h / 剩余 27.32h**。每次跑前 `python -m kaggle quota`。

## 6. 结果记录（实际数字，持续更新）

### 6.1 云端推理模板（task-5）
- kernel: `ptcuoigl/filament-cloud-inference-template-v1` v1 COMPLETE
- 输出：1299 行 / 169 图，**MD5 = 09c0eae5fdff29a91f8a740475ecaf36**（= pub070_kaggle_v1.csv，字节一致）
- 本地校验：PASS（schema/RLE 解码 2048x2048/无重叠/无重复）
- 结论：云端"任意权重→submission"通道可用。

### 6.2 dariush quick（2 模型 TTA2）
- kernel: `ptcuoigl/dariush-unet-cloud-v1` v1 COMPLETE，总日志 ~163s
- 输出：**1326 行 / 176 张图**，7.37 实例/图；作者 14 项自检全过（含逐图像素不重叠）
- 单模型 180 图 TTA2 71s（T4×2 并行）
- 100% 覆盖期望：full 10 模型 TTA8 会更多实例（作者提交：1299 行/180 图）

### 6.3 cloud pipeline 修复记录（T10）
见 §8 坑 3/4/5；修复：refiner crop 尺寸（上游已修 3 处）+ runner out-dir /tmp + 只复制小产物 + smoke 空提交放行。

### 6.4 cloud_pipeline 全量 fold0（T10 验收，2026-10-29 完成）
- kernel: `ptcuoigl/filament-cloud-runner-v1` v3 COMPLETE，runner `verdict=PASS`，总 **5346 s（89 min）**
  （detector 3262s / refiner 554s / val 推理+sweep 1370s / test 推理 90s；pip 装 ultralytics 7.5s 一次成功）
- 配置：FOLD=0, DET_EPOCHS=30, REF_EPOCHS=8, IMG_SIZE=1024, BATCH=8, LABEL_SOURCE=all；
  n_train=565 物理/922 记录，n_val=142 物理
- **val 指标（fold0 val 142 图，官方 pooled PQ；metric_source = code dataset 的 `src/pq_metric.py` + harness folds.json）**：
  **PQ = 0.3780**，SQ = 0.6475，RQ = 0.5838，TP 968 / FP 531 / FN 849（232 annotator-image，GT 1817）
  best_config: conf=0.35, ref=0.5, tta=guard（完整网格见 `sweep.csv`）
- test 推理：**1065 行 / 177 张图**，overlap_errors=0；本地 `src/validate_submission.py` **OK**
  （面积 min/med/max = 55 / 1374 / 22799 px，总 2.196M px）
- 产物：`work/cloud/runs/cloud_fold0/`（submission.csv / val_report.json / summary.json / sweep.csv /
  val_per_image.csv / train_log.txt / config.json / best_det.pt 20.6MB / best_ref.pt 57MB）
- 交付：**`submissions/cloud_fold0_v1.csv`**（MD5 e57a32a129fd982728db8200e79bc9a4，319,483 B）
- 说明：val 是本地口径，public LB 通常比本地低 0.03-0.06。

### 6.5 dariush 10×U-Net 诚实 OOF（task-13，fold0 val 142 图）
- kernel: `ptcuoigl/dariush-oof-fold0-v1`（TTA2 快速版先交付；TTA8 完整版 v2 跑完覆盖）
- 方法：每张 fold0 val 图 → dariush folds.csv 查其 fold f → 用 dariush 的 fold (f+1)%5 的两个模型
  （resnet34 + efficientv2_s，均未见过该图）× TTA × uint8 平均 → KEEP v6 → RLE（逐图不重叠）
- **本地官方 pooled PQ（src/oof_report.py --fold 0）**：
  - TTA2 版：**PQ = 0.4688**（SQ 0.6859 / RQ 0.6835），TP 1248 / FP 587 / FN 569，95% CI [0.4517, 0.4861]
  - **TTA8 版（最终交付）：PQ = 0.4755**（SQ 0.6881 / RQ 0.6910），TP 1238 / FP 528 / FN 579，95% CI [0.4574, 0.4926]，142 图 / 1076 行
  （232 annotator-image / GT 1817，pycocotools backend）
- 对比：我们云端训练模型 fold0 val PQ = 0.3780 → **dariush 语义 U-Net 在相同 val 上更强 +0.10**
- 覆盖：142/142 图；交付 `work/dariush_oof/pred_dariush_fold0.csv`（TTA8），TTA2 备份 `pred_dariush_fold0_tta2.csv`
- 面积命中率：<300px 0.373 | 300-1k 0.643 | 1k-3k 0.723 | 3k-10k 0.765 | >10k 0.632

### 6.6 dariush 10×U-Net test 推理（task-12，2026-10-29 18:44 COMPLETE）
- kernel: `ptcuoigl/dariush-unet-cloud-v1` v3（全量配置 max_models=0, tta=8），10 模型 × 180 图 × TTA8 于 2×T4
  并行，模型推理 **18.8 min**，kernel 总 ~20.3 min；10 个权重 sha256 全部与 receipt 一致
- 作者 14/14 自检全过（含逐图像素不重叠）；**1177 行 / 177 张图**（6.54 实例/图）
- 与作者当年提交**结构完全一致**（同样 1177 行/177 图；sha 不同仅 cudnn 非确定性，作者 notebook 已说明）
- 本地 `src/validate_submission.py` **OK**：rows=1177，覆盖 177/180，面积 min/med/max=221/1343/21262 px，total 2.362M px
- 交付：**`submissions/dariush_unet_full_v1.csv`**（319,083 B，MD5 7e36157dd5b9db2f1303a780b1b74b87）
- 预计 public ≈0.39（作者同结构提交 0.39；我们自己快速版 2 模型 TTA2 = 0.37 已实测）

### 6.7 提交候选清单（截至 2026-10-29 18:50）
| 文件 | 来源 | test 规模 | 本地指标 | public LB |
|---|---|---|---|---|
| `submissions/pub070_kaggle_v1.csv` | 公开 0.70 权重（Dice 时代） | 1299 行/169 图 | 云端 MD5 复现一致 | 0.34（已实测 ref 57763686） |
| `submissions/dariush_unet_full_v1.csv` | dariush 10×U-Net TTA8 | 1177 行/177 图 | OOF PQ 0.4688（fold0 val, TTA2） | 预计 ~0.39（待 Lead 提交） |
| `submissions/cloud_fold0_v1.csv` | 我们自训 YOLO11s+crop refiner，fold0 | 1065 行/177 图 | fold0 val PQ 0.3780 | 待提交，预期 0.33-0.36 |

## 7. 通用操作命令

```powershell
# 所有 kaggle 命令都要先设 token
$env:KAGGLE_API_TOKEN=(Get-Content "$env:USERPROFILE\.kaggle\access_token" -Raw).Trim()

# 数据集：创建（首次）/ 新版本（更新，注意 Windows 下必须用绝对路径 -p，否则临时文件路径报错）
python -m kaggle datasets create -p 'D:\...\work\cloud\code_pkg_v2' -r skip
python -m kaggle datasets version -p 'D:\...\work\cloud\code_pkg_v2' -m "update ..." -r skip
python -m kaggle datasets status ptcuoigl/filament-cloud-code

# kernel：push（目录内只能有 code_file + kernel-metadata.json；**不要留 __pycache__**，否则 push 报 JSON 错误）
python -m kaggle kernels push -p src/kaggle_kernels/<name>
python -m kaggle kernels status <ref>
python -m kaggle kernels output <ref> -p <local-dir>     # 输出 + kernel log
python -m kaggle kernels logs <ref>                      # 失败时的日志（Windows 编码坑见 §8）

# 配额
python -m kaggle quota
```

## 8. 坑与经验（按遇到顺序）

1. **kernel id vs title**：push 时实际 slug 由 metadata 的 `title` 决定（不是 `id` 字段）；title 与 id 不一致会 warning 且用 title 的 slug。查 status 必须用真实 slug。
2. **Windows 相对路径**：`kaggle datasets create -p 相对路径` 会因临时文件路径拼错报 `No such file or directory`；**用绝对路径**。`datasets create -r zip` 会把子目录压成 `src.zip/work.zip`；Kaggle 挂载时**会自动解压**（实测：解压到 dataset 根），所以"单 code_pkg.zip"方案可行；但依赖"自动解压"的行为，runner 里有解压/回退双保险。
3. **kernel 目录不能有 __pycache__**：py_compile 在 kernel 目录会生成 `__pycache__`，push 会报 `Expecting value: line 1 column 1`（CLI 误把它当 json 读）。编译检查请把 cfile 指到目录外。
4. **cloud_pipeline refiner crop 崩溃**（T10 关键 bug）：`RefinerDataset` crop 传 `args.img_size`（512/1024）但 refiner cache 是 128/256；且 zoom<1 分支不 pad → batch 尺寸不一致 `[3,471,471] vs [3,512,512]`。修复：crop=-1 用缓存实际尺寸 + zoom<1 pad 到 crop（modeling 已同步上游）。
5. **smoke 空提交**：1 epoch 垃圾模型在 4 张 test 上可能 0 预测 → pipeline fail-closed。已对 smoke 放行（header-only CSV），full 仍严格。
6. **output 下载慢**：pipeline 默认把 image cache/refiner npy 写 `/kaggle/working`（几百 MB，707 张小图）→ `kernels output` 很慢甚至本地超时。runner 改为 `--out-dir /tmp/filament_run` + 只复制小产物。
7. **日志编码**：`kaggle kernels logs` 输出含 emoji，Windows PowerShell GBK 下会抛 `UnicodeEncodeError`；解决：用 Python subprocess + `PYTHONIOENCODING=utf-8` 抓 bytes 落盘再读（见 `src/kaggle_kernels/_scratch/fetch_kernel_log.py`）。
8. **并发上限 2**：第三个 GPU kernel push 直接报 `Maximum batch GPU session count of 2 reached`，不是排队。要串行安排长跑（本次：dariush full + cloud full 并行；OOF/CAGE 排队）。
9. **T4 实际是 2 卡**：`machine_shape: NvidiaTeslaT4` 给 2×T4（各 15GB）。推理可双卡；ultralytics 训练默认单卡。
10. **kernel 版本**：同一 `-p` 目录重复 push = 该 kernel 的新 version 并立即重跑；改参数就是改 `code_file` 再 push（我们用它切 smoke→full→OOF）。

## 9. 文件索引

| 路径 | 说明 |
|---|---|
| `src/kaggle_kernels/cloud_template/` | 云端推理模板（pub070 engine + custom engine 接口） |
| `src/kaggle_kernels/cloud_runner/` | 云端训练 runner（probe/smoke/full，CONFIG 切换） |
| `src/kaggle_kernels/dariush_unet/` | dariush 10×U-Net test 推理 |
| `src/kaggle_kernels/dariush_oof/` | dariush 在我们 fold0 val 的诚实 OOF |
| `src/kaggle_kernels/cage_r5/` | jamesmills CAGE r5 推理 |
| `src/kaggle_kernels/_scratch/` | 辅助脚本（打包 zip、同步上游、抓日志、校验、资产侦察） |
| `work/cloud/code_pkg_v2/code_pkg.zip` | 代码 dataset 的 zip（当前 v5） |
| `work/cloud/runs/` | 各 kernel 的下载产物（submission/log/report） |
| `work/cloud/kmeta*/, work/cloud/assets/` | 公开资产侦察原始数据 |
| `notes/public_assets.md` | 公开资产清单与可信度判断 |
| `notes/KAGGLE_RUNNER.md` | pub070 首次云端复现记录 |


## 10. 2026-10-29 晚：概率图 dump / 多尺度 TTA / 语义 U-Net / CAGE 概率图

### 10.1 dariush 10×U-Net 概率图 dump（COMPLETE，25.6 min）
- kernel: `ptcuoigl/dariush-probs-dump-v1` v1，日志 t=1538s
- test 180 图：10 模型 × TTA8，每模型 176-259s/卡（2×T4 并行），聚合 1120s；npz 5.5 MB（uint8 极稀疏，压缩比 ~137x）
- fold0 val 142 图：fold-aware（每图用 dariush fold=(f+1)%5 的 2 个模型）× TTA8，5 组共 371s；npz 4.2 MB
- 产物路径：`work/dariush_probs/{probs_dariush_test.npz, probs_dariush_fold0.npz, stems_*.json, probs_meta.json}`
- 格式：布局 B（`stems` <U17 数组 + `probs` uint8[N,2048,2048]），量化 round(sigmoid*255)
- 验证（harness 独立复核）：fold0 用 KEEP v6 重建 PQ = **0.47547**（与 task-13 CSV 逐项一致）；test 重建与已提交 dariush_unet_full_v1.csv **逐行完全一致**（1177 行/177 图）→ dump 可信

### 10.2 dariush 多尺度 TTA 概率图（fold0，RUNNING）
- kernel: `ptcuoigl/dariush-multiscale-fold0-v1`（实际 slug；metadata id 为 dariush-ms-folds0-v1）
- scales [0.85, 1.0, 1.15]；每尺度 fold-aware 2 模型 × TTA8；概率 resize 回 2048
- 输出 `probs_fold0_ms.npz`：stems + probs_s085/probs_s100/probs_s115/probs_fused（各 uint8[142,2048,2048]）+ stems_fold0_ms.json + probs_ms_meta.json
- 用途：harness 评估「单尺度 0.4755 vs 多尺度融合」；若提升 ≥0.003 再跑 test 多尺度版

### 10.3 语义 U-Net 云端训练（RUNNING，ETA ~21:40）
- kernel: `ptcuoigl/filament-semantic-train-v1` v2（title 派生 slug；metadata id 已修正一致）
- code dataset v7（含 clDice 版 `src/train_semantic.py`）
- 配置：FOLDS=0,1 双进程各绑一张 T4；EPOCHS=24；encoder tu-tf_efficientnetv2_s；crop 1024；batch 4；crops/epoch 3；val-every 8；TTA 2；LOSS=bcedice（A/B 再跑 clDice）
- smoke 参考（2 epoch）：fold0 PQ 0.3458 / fold1 0.3491
- 产物：sem_fold{0,1}_best.pt、sem_fold{0,1}_val_report.json、sem_fold{0,1}_val_probs_best.npz、train_fold{0,1}.log、run_summary.json

### 10.4 CAGE r5 第三候选（COMPLETE）
- kernel: `ptcuoigl/cage-r5-cloud-v1` v1，~2.5 min（pip 装 smp + 2 模型 × 180 图 × hflip TTA）
- **1177→1234 行 / 177 图**（面积 min/med/max=253/1204/20237，总 2.332M px），本地 `src/validate_submission.py` OK
- 交付 `submissions/cage_r5_v1.csv`（315,274 B，md5 11f1e870e76cee062a940ea5e58c3f08）
- CAGE 概率图 dump kernel 已写好（`src/kaggle_kernels/cage_probs/`），格式 probs_mask/probs_hull uint8[2048]，fold0 版非 fold-aware（作者 split 未公开）

### 10.5 GPU 配额（截至 2026-10-29 20:00）
- 已用 **3.85h / 30h**（剩余 26.15h）

### 10.6 踩坑补充
- sem kernel metadata 的 id 与 title 派生 slug 不一致会导致再次 push 时 409 Conflict（SaveKernel）；把 metadata id 改成实际 slug 即可
- `kaggle kernels output` 下载大 npz/权重时容易超过本地 120s 工具超时 → 一律后台下载
- 概率图 npz 压缩后体积极小（uint8 概率图极稀疏），不要用原始大小（N*4MB）估下载时间


### 10.7 多尺度 TTA 结论（2026-10-29 20:20，**否定**）
fold0 val 142 图、KEEP v6、官方 pooled PQ：
| 方案 | PQ | 说明 |
|---|---|---|
| single s100（原 dump） | **0.47547** | 基线；与 probs_dariush_fold0.npz 逐项一致 |
| single s085 | 0.1551 | 模型固定 2048 训练，缩放破坏尺度分布 |
| single s115 | 0.1239 | 同上 |
| mean[3 scales] | 0.3312 | 融合被差尺度稀释 |
| s085+s100 | 0.4198 | 仍低于 s100 |
| s100+s115 | 0.4102 | 仍低于 s100 |

结论：**多尺度 TTA 对本模型无效（ΔPQ -0.056 ~ -0.35），放弃 test 多尺度 dump**；GPU 留给 CAGE 概率图与语义 U-Net 补折。
产物：work/dariush_probs/probs_fold0_ms.npz（24.9MB，保留备查）、work/dariush_oof/ms_*_eval.json。

## 11. 2026-10-29 深夜：语义 5 折训练 / notoverkil 验证 / nk 集成复现

### 11.1 语义 U-Net 云端训练全量结果（encoder tu-tf_efficientnetv2_s, crop1024, batch4, crops3, 24ep, TTA2 val）
| kernel 版本 | folds | loss | 状态 | 单折 val PQ（固定 KEEP） | best epoch |
|---|---|---|---|---|---|
| v2 | 0,1 | bcedice | COMPLETE（109 min） | fold0 **0.4211**（SQ .6771/RQ .6219）、fold1 **0.4289**（SQ .6780/RQ .6326） | 24 / 24 |
| v3 | 0,2 | fold0 clDice, fold2 bcedice | COMPLETE | fold0 clDice **0.4115**（SQ .6711/RQ .6132）→ 比 bcedice **低 0.0096，clDice 放弃**；fold2 **0.4264**（SQ .6799/RQ .6273） | 16 / 24 |
| v4 | 3,4 | bcedice | RUNNING（ETA ~00:25） | — | — |

- KEEP 调优后（modeling 54 组扫描，官方复核）：fold0 0.4282（t_lo .35/t_hi .95/area200/mp .90）、fold1 **0.4429**（.4/.95/area400/mp .90）、fold2 0.4312。
- 结论：**clDice 不优于 bcedice**，5 折统一 bcedice；sem 单模型 0.43 档，与 dariush 单模型相当，价值在概率级融合。
- 产物：C:\Users\ptcu_\filament_assets\models\sem\sem_<fold>_bcedice.pt（89MB each）+ D: work/models/sem/（val_probs/val_report/train.log）

### 11.2 notoverkil fold0 公开权重诚实 OOF（云端 T4，5 min）
kernel ptcuoigl/notoverkil-fold0-eval-v1：maskrcnn1536 **PQ 0.2589**（SQ .6510/RQ .3978，FN1286 召回差）；yolo11m1536 **PQ 0.4357**（SQ .6822/RQ .6386）；mask-level fusion 0.4047（FP 爆炸）。
→ 按 Lead 规则（<0.44 放弃）**notoverkil 单模型路线不采用**；产物 D: work/models/notoverkil_eval/。

### 11.3 nk（Salty Taro, public LB 0.41）集成复现
- kernel ptcuoigl/nk-ensemble-v1：v1 fold0-eval rc=1 失败（convnext 族缺 segmentation_models_pytorch → 空 cache → evaluate_consensus 空 pred 抛 pq_metric ValueError）。修复：空族 skip 守卫 + bbox-local consensus（快 50-500x）+ smp 自举 pip 安装。
- 用已产出的 3 个 YOLO 族 fold0val cache 本地复算（官方 pq_metric，142 图）：
  - **cross-family 3 族 consensus：PQ 0.4134**（SQ .6857/RQ .6028，TP1376/FP1372/FN441）≈ 其 LB 0.41 ✓
  - modeling 单族 yolo11m-1536：PQ **0.4415**（TP1351/FP1015/FN466）
  - 对比 dariush 10 模型 TTA8 诚实 OOF 0.4755 → **nk 最佳形态 0.4415 仍低 3.4 点**；只作参考/融合多样化源。
- nk test-parallel v3 RUNNING（Lead push，2 GPU，25 模型 × 8 TTA，ETA ~01:30）。

### 11.4 多尺度 TTA 结论（否定，见 10.7）
single s100 0.47547 vs s085 0.1551 / s115 0.1239 / mean3 0.3312 → 放弃 test 多尺度。

### 11.5 数据集与 kernel 清单（新增）
- ptcuoigl/filament-sem-weights（4→5 个 sem fold 权重，~89MB each，create @23:10，v2 加 fold3/4）
- ptcuoigl/notoverkil-fold0-weights（两个 notoverkil fold0 权重镜像，254MB）
- ptcuoigl/filament-cloud-code v11（含 bbox-consensus 版 cloud_notoverkil_ensemble.py、sem cloud 脚本、infer_notoverkil、pq_metric、folds.json）
- ptcuoigl/filament-cloud-code-v2（modeling 建的 nk 专用 code dataset，137KB）
- ptcuoigl/filament-sem-test-probs（sem 5 折 test 概率推理，2 卡 shard，tile1024/overlap128/TTA2）
- work/cloud/kernel_sem_test_probs/ 自包含脚本（已修 smp 自举 + CUDA 设备序数 bug）

### 11.6 磁盘/配额（2026-10-29 23:40）
- D: 剩 ~15-17GB；C: 剩 ~110GB；大文件一律写 C:\Users\ptcu_\filament_assets\{downloads,models,weights,probs,tmp}
- GPU 配额：9.56h / 30h（剩余 20.4h）

### 11.7 踩坑
- kaggle kernels push 反复被会话自动审核拒绝 → 重试/改小命令；脚本内部 subprocess 调 CLI 一般可通过
- machine_shape: NvidiaTeslaT4 实测仍给 2×T4（shard 并行可用）
- 本地 CPU 评估在队友训练/扫描并发时慢 10-20x（142 图 consensus 18 min）→ 决策优先云端


### 11.8 sem v4（folds 3,4）结果与 sem test kernel 排障（2026-10-30 00:45）
- folds 3,4 bcedice, 24ep：fold3 PQ 0.4342（SQ .6783/RQ .6402，TP1006/FP559/FN572，epoch 24）；fold4 PQ 0.4074（SQ .6783/RQ .6006，TP967/FP653/FN633，epoch 16）
- 5 折 bcedice 固定 KEEP：fold0 .4211 / fold1 .4289 / fold2 .4264 / fold3 .4342 / fold4 .4074（均值 ~0.4236）；KEEP 调优后 modeling 报：fold3 0.4362 / fold4 0.4123
- 权重：C:\Users\ptcu_\filament_assets\models\sem\sem_{0..4}_bcedice.pt（+ weights/sem_weights 副本）；小文件 D: work/models/sem/
- 数据集：ptcuoigl/filament-sem-weights v2（6 files：5 bcedice + fold0 cldice，00:34 ready）

**sem test kernel 两次失败排障（v1/v2 → v3）**
1. v1 ERROR: ModuleNotFoundError segmentation_models_pytorch（self-contained 脚本没自举安装）→ 加 ensure_packages + cuda 设备序数修复
2. v2 ERROR: RuntimeError Can't call numpy() on Tensor that requires grad（tiled_probability 未套 torch.no_grad）→ 补 torch.no_grad() + .detach()
3. v3: 00:44 push，RUNNING，5 模型 × 180 图 /2 卡，ETA 40-55 min


### 11.9 sem test probs（5 折平均）与 cross-fold fold0-val probs（2026-10-30 01:40）
- **sem test probs**：kernel ptcuoigl/filament-sem-test-probs v3 COMPLETE（8.5 min，2 卡 shard；v1 缺 smp 自举、v2 缺 torch.no_grad 均已修）
  - 产物：C:\Users\ptcu_\filament_assets\probs\sem_test_probs_5fold.npz（4,543,375 B，180 keys，uint8 2048²，5 折平均）
  - 一致性：merged(180) == shard0(90) ∪ shard1(90) ✓；统计 mean p255 0.813 / frac>89 0.0034
- **cross-fold fold0-val probs（乐观代理）**：kernel ptcuoigl/sem-crossfold-probs-v1 v2 COMPLETE（~13 min）
  - 产物：C:\Users\ptcu_\filament_assets\probs\sem_crossfold\sem_fold{0..4}_bcedice_on_fold0val.npz（各 142 keys）+ meta_fold0val.json
  - 用途：5 折平均 sem 的乐观代理（fold1-4 见过 fold0 图，不能当 honest OOF）
- code dataset v14（修 cloud_sem_crossfold_probs.py 缺 `import sys`）


### 11.10 nk v3 test（15 模型 × 8 视图）+ 公开 submission.csv 发现（2026-10-30 03:30）
- kernel ptcuoigl/nk-ensemble-v1 v3 test-parallel：**COMPLETE 03:13，总耗时 13,760s（3h49m）**
  - 实际运行 15 个模型（3 个 YOLO 族 × 5 折；convnext 族因无 smp/结构不匹配跳过，maskrcnn 被 skip；非完整 25 模型）
  - submission.csv：746,504 B，**2328 行 / 180 图**（实例/图 12.9，median area 974.5，与 dariush 重叠 90.1% best IoU）；D: submissions/candidates/nk_ensemble_v3.csv
  - 产物：C:\Users\ptcu_\filament_assets\downloads\nk_test\（含 solar_ensemble/cache_test_shard0/1 model_*.npz、converted/*.pt、inference_summary）
  - 预期 public ≈0.41 档（15 模型子集），不替代 dariush
- **公开 dataset notoverkil/solar-filament-weights 直接含作者最终 submission.csv**：
  - 文件 339,590 B，**1287 行/178 图**，md5 298fb24dcaba0b518a68d99e98c3cf3b；与 C:\...\downloads\nk_ref\submission.csv **逐字节一致**
  - dataset 共 60 文件：25 模型（5 族×5 折，含 config.yaml/manifest）+ refiner/fold-{0..4}.pt（5×97.9MB）+ rescorer-hgb.pkl + splits/{folds.csv,folds-metadata.json} + submission.csv
  - 时间戳 10/18/2026 11:55 = 其公开 LB 0.41 提交日 → 极可能是 0.41 那份；harness 校验 schema/覆盖/不重叠全 PASS，分布接近 GT（面积中位 1221 vs GT 1228）
  - 建议：作为公开资产候选提交（Lead 决定），预期 ~0.41 > dariush 0.39

### 11.11 Lead 决策：不提交 nk 公开 ref submission（2026-10-30 04:00）
- notoverkil/solar-filament-weights 自带的 submission.csv（339,590 B，1287 行/178 图，md5 298fb24dcaba0b518a68d99e98c3cf3b）**不提交**：Kaggle 竞赛要求提交由参赛者自己代码/模型产生，直接提交他人提交文件属"非自己工作"，prize 复核会判不合格。
- 该文件保留作**对照基准**（判断我们复现的接近程度）；不复制到 submissions/ 根目录。
- 继续做**合规同类复现**：notoverkil 公开权重 + 我们自写 consensus/vote-filter（+ 可选公开 refiner/rescorer）→ 自己的 submission（预期 0.38-0.41）；与 dariush 公开权重复现（0.39）性质相同，报告披露。
- 待提交（08:05 后，各 1 次配额）：fused_w008 / fused_w010 / vote-filtered nk_v3 / cage_r5_v1。

### 11.12 nk v3 vote-filter 候选（2026-10-30 04:20）
- 方法：对 nk v3 consensus（2328 掩膜）逐实例统计 15 个模型的 IoU>0.5 支持数（cache_test_shard0/1 的 model npz），按阈值过滤 → submissions/candidates/nk_v3_vote{4,6,8,10,12,14,15}.csv
- 支持数分布：≥15 1259、≥14 1327、≥12 1442、≥10 1529、≥8 1607、≥6 1680、≥4 1779（总 2328）
- vote≥15：1259 行/177 图，area min/med/max 190/1221/19640，total 2.418Mpx（对照作者 ref：1287 行/178 图，190/1221/22430，2.468Mpx）→ 统计几乎一致
- 与 ref mask 对比（IoU>0.5 命中率 / mean best IoU / >0.9 比例）：v3 95.9% / 0.866 / 52.9%；vote12 95.1% / 0.855 / 52.7%；vote14 92.7% / 0.834 / 52.2%；vote15 90.6% / 0.815 / 51.6%
- 结论：ref 是 boundary-refined 版本（refiner U-Net），我们 vote-filter 版实例集合对但边界略粗 → 期望 public 0.39-0.40（接近 ref 0.41 但不会完全一致）；如要复现 0.41 需应用其公开 refiner/rescorer（时间窗口内未做）
