# moonboost 差距矩阵（逐维度对比现有项目）

规则：每格写实测数字或“未测”；只写本机/同会话测得的数，第三方数字标注来源。✅ 领先 / ≈ 持平(±1%) / ❌ 落后 / ? 未测。

## A. 准确率（GBM-perf airline 1M→100k，100 轮，lr 0.1，单线程，2026-11-18 专用机）

| 配置 | AUC | 时间 |
|---|---|---|
| moonboost（标签编码，depth10） | 0.7537 | 3.9 s |
| XGBoost 3.4.1 hist 标签编码 | 0.7533 | 7.9 s |
| XGBoost 3.4.1 原生类别（max_cat_to_onehot=1） | 0.7603 | 29.8 s |
| LightGBM 4.7.0 depth10 标签编码 | 0.7545 | 6.8 s |
| LightGBM 4.7.0 depth10 原生类别 | 0.7607 | 7.5 s |
| LightGBM 4.7.0 原生类别 num_leaves=512 | **0.7650** | 7.9 s |
| CatBoost depth10 无类别 | 0.7282 | 10.3 s |
| CatBoost depth10 原生类别 | 0.7330 | 33.0 s |

当前：moonboost lossguide 4096 叶 + 类别 0.7878；LightGBM 最佳（oh32、3072 叶）0.7825；XGBoost 原生类别 0.7603。领先 0.005。计时待单独复测。

## B. 维度清单

| 维度 | moonboost | 最强对手 | 状态 |
|---|---|---|---|
| 准确率 airline（同叶数 255/1023/4095） | .7629/.7779/.7868 | LightGBM .7612/.7761/.7829 | ✅ |
| 准确率 OpenML 10 个二分类（5 次重复划分 × 12 组等预算调参，R2 含采样） | 对 LightGBM：amazon +.0053、electricity +.0047、phoneme +.0052（均 >2.5 se），其余持平；对 XGBoost：10 个全部持平 | XGBoost/LightGBM | 对 LGB 部分✅，对 XGB ≈ |
| 单线程训练速度 GBM-perf 1M | 3.7–3.8 s（11-23 同会话；logistic/softmax 梯度 v128 SIMD） | LightGBM 6.2 s，XGBoost 7.8–7.9 s | ✅ |
| 多线程训练速度 | 不支持（MoonBit 无线程） | XGBoost 16 核 2.7 s（GBM-perf 官方，不同硬件） | ❌ |
| 推理速度 native vs XGBoost C API | 见 README | | ✅（README） |
| 浏览器推理 | 见 README | onnxruntime-web | ✅（README） |
| 浏览器训练 | wasm-gc 0.30 s（100k×30） | ml-xgboost 1.01 s | ✅ |
| 推理逐位一致 | XGBoost、LightGBM | 无人做 | ✅ |
| 形式化验证 | 有（moon prove 内核） | 无 | ✅ |
| 包体积 JS | 270 KB / 77 KB gz | ort-web 14.8 MB wasm | ✅ |
| 类别特征训练 | 有（one-vs-rest + 排序划分，XGBoost 格式） | LightGBM/XGBoost/CatBoost | ✅（airline 同叶数更准） |
| 按叶生长 | 有（lossguide，叶堆） | LightGBM | ≈/✅（11-21 核 2 同会话，带类别：255 叶 6.4 vs 6.5 s；1023 叶 8.8 vs 9.2 s；AUC 均更高；4095 见 log） |
| 训练目标函数种类 | 19 种（已覆盖 XGBoost 全部 19 个 objective）（11-23 rank:map 对 XGB MAP +0.0024 显著、survival:aft 1.4–2.4× 快；11-21 加 absoluteerror、quantileerror；11-21 rank:ndcg/pairwise；11-22 binary:hinge、survival:cox，速度快于 XGBoost、精度持平；分位数叶；XGBoost 加载逐位一致；10 万行 4 种子：三项均优于 XGBoost 约 0.5%，比 LightGBM 差 0.04–0.13%（约 1 se）） | reg:logistic 以外的 multi:… 等 | ❌（缩小） |
| 行/列采样（subsample、colsample_bytree/bynode） | 有；11-20 起只在保留行上建树。airline 带类别 sub 0.5：255 叶 4.0 s vs LGB 11.3 / XGB 22.2，1023 叶 6.1 vs 14.3 / 33.6，AUC 均最高 | XGBoost/LightGBM | 采样训练速度 ✅（单线程）；功能 ≈ |
| min_data_in_leaf（按 hessian 估计行数，同 LightGBM） | 有（11-19 加） | LightGBM；XGBoost 无 | ≈ |
| 梯度采样（GOSS/MVS） | 有 MVS（sampling_method=gradient_based，CPU；同采样率优于均匀采样，0.5 时 4 个数据集 +.003～.0075） | LightGBM GOSS、CatBoost MVS；XGBoost 仅 GPU | ≈（对 XGBoost CPU ✅）；11-20 起未采样行不进直方图，阈值两遍精确解，比旧版快 28% |
| 高基数类别（>255 类） | 覆盖不足 99% 时保留全部类别（11-19） | XGBoost 全保留 | ≈ |
| GPU | 无 | XGBoost/CatBoost | ❌（非目标） |
| 峰值内存（11-24，bench/peak_memory.py，整进程） | 类别 255/1023/4095 叶 307/337/437 MB；GBM-perf 128 MB | LGB 320/378/621、343（Python 基线 257/227 MB）；XGB 477/625/1113、409 | GBM-perf、4095 叶 ✅；255/1023 叶扣除 Python 基线后 ❌（落后 LGB，6 级复制约 176 MB） |


## 训练速度 airline+类别（单核）
11-20 空闲机（S7 后，air_time.py 核 2）：255 叶 mb 7.2 s / lgb 7.0；1023 叶 11.1 / 9.9 / xgb 36.5；4095 叶 19.7 / 20.3 / 64.6。

旧（S3 后）：
255 叶 mb 6.8 s / lgb 6.5 / xgb 20.4；1023 叶 10.7 / 9.6 / 37.1；4095 叶 19.7 / 19.6 / 56.2。

| 排序（MSLR-WEB10K，单线程） | 速度 40–41 s vs LGB 42.9 / XGB 60.5；NDCG@10 .5051 vs .5042/.5042（不显著） | 持平 LGB，优于 XGB 速度 |
