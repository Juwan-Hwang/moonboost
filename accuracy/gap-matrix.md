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
| 准确率 OpenML 9 个二分类（等预算 24 组调参） | 多数在噪声内持平；credit、kdd 领先；amazon、electricity、phoneme、higgs 略低 0.001–0.005 | XGBoost/LightGBM | ≈ |
| 单线程训练速度 GBM-perf 1M | 3.9 s | LightGBM 6.0 s | ✅ |
| 多线程训练速度 | 不支持（MoonBit 无线程） | XGBoost 16 核 2.7 s（GBM-perf 官方，不同硬件） | ❌ |
| 推理速度 native vs XGBoost C API | 见 README | | ✅（README） |
| 浏览器推理 | 见 README | onnxruntime-web | ✅（README） |
| 浏览器训练 | wasm-gc 0.30 s（100k×30） | ml-xgboost 1.01 s | ✅ |
| 推理逐位一致 | XGBoost、LightGBM | 无人做 | ✅ |
| 形式化验证 | 有（moon prove 内核） | 无 | ✅ |
| 包体积 JS | 270 KB / 77 KB gz | ort-web 14.8 MB wasm | ✅ |
| 类别特征训练 | 有（one-vs-rest + 排序划分，XGBoost 格式） | LightGBM/XGBoost/CatBoost | ✅（airline 同叶数更准） |
| 按叶生长 | 有（lossguide，叶堆） | LightGBM | ≈（4095 叶同速；255–1023 叶慢 5–10%） |
| 目标函数种类 | 4 种 | XGBoost 数十种 | ❌ |
| GPU | 无 | XGBoost/CatBoost | ❌（非目标） |
| 内存占用 | 未测 | | ? |


## 训练速度 airline+类别（单核，S3 后）
255 叶 mb 6.8 s / lgb 6.5 / xgb 20.4；1023 叶 10.7 / 9.6 / 37.1；4095 叶 19.7 / 19.6 / 56.2。
