# 准确率 RSI 记录（GBM-perf airline，100 轮，lr 0.1，单线程；AUC 为 100k 测试集）

噪声：同类超参之间 ±0.003 的波动常见（单次运行，非种子噪声），小于 0.003 的差异不作结论。
测试集是训练集之后年份的航班（时间外推），过拟合类别的细节会伤害泛化。

| # | 假设 | 结果 | 结论 |
|---|---|---|---|
| P1 | LightGBM 原生类别 vs 标签编码 | 0.7607 vs 0.7545（depth10） | 类别切分 +0.006 |
| P2 | 按叶生长 512 叶 | 0.7650 | +0.004 |
| P3 | 加 DepTime 的小时、分钟特征 | 0.7665 | +0.0015（边缘） |
| P4 | 5 折 OOF 目标编码 8 组类别交叉 | 0.7533 | 否决：-0.013，时间外推下目标编码过拟合 |
| P5 | cat_smooth / min_data_in_leaf / lambda_l2 扫描 | 0.7621–0.7675 | 噪声范围内，无可靠收益 |
| P6 | 叶数 1024 / 2048 / 4096（不限深度） | 0.7712 / 0.7716 / 0.7694 | 树规模是最大杠杆，峰值 1–2k 叶 |
| R1 | 训练后对所有树叶值联合回拟（循环 Newton，λ=10/1000/30000） | 0.7329 / 0.7507 / 0.7198 | 否决：严重过拟合或欠拟合 |
| P7 | LightGBM 类别 one-vs-rest 阈值 max_cat_to_onehot 8/16/32/64/512 | 0.7700/0.7702/0.7760/0.7760/0.7719 | ≤32~64 类走 one-vs-rest 最好（DayofMonth 31 类关键） |
| P8 | LightGBM onehot32 + 2048 叶 | 0.7781 | 调优后 LightGBM 最佳 |
| N1 | 原型：类别排序用偶数行、增益用奇数行（honest 交叉拟合） | oh32 1024 叶 0.7711→0.7738；oh4 0.7640→0.7711；oh32 2048 叶 0.7803→0.7819 | 一致正向，待移植 |
| P9 | 原型 leaf_newton（叶值多步 Newton） | 0.7635→0.7623；2048 叶 0.7709→0.7693 | 否决 |
| M1 | **moonboost 实装**：类别切分（one-vs-rest ≤ max_cat_to_onehot，否则按 g/(h+10) 排序前后缀，cat_l2=10，≥100 行）+ lossguide 按叶生长 | depthwise depth10 无类别 0.7520 / 有类别 0.7629；lossguide 1024 叶 0.7785；**2048 叶 0.7855** | 超过 LightGBM 最佳 0.7781；XGBoost 加载同一 JSON 预测逐位一致 |
| M2 | moonboost lossguide 叶数 3072 / 4096（oh32） | 0.7873 / 0.7878 | 更大树继续涨 |
| M3 | moonboost oh64（2048 叶） / min_child_weight 5 | 0.7858 / 0.7833 | oh64≈oh32；mcw5 变差 |
| M4 | moonboost cat_crossfit（N1 移植，2048/4096 叶） | 0.7853 / 0.7866 vs 0.7855 / 0.7878 | 否决：原型收益未复现，且慢 60% |
| P10 | LightGBM oh32 3072 / 4096 叶；同 moonboost 正则（λ=1、min_sum_hessian=1、min_data=1）2048/4096 | 0.7825 / 0.7823；0.7780 / 0.7803 | LightGBM 最佳 0.7825，moonboost 4096 叶 0.7878 领先 0.005 |
| M5 | 多数据集（OpenML 10 个二分类，80/20，100 轮 lr0.1，每库 4 组配置按验证集选） | 胜：credit 0.832/0.817/0.790、kdd_upselling 0.855/0.848/0.840、higgs 0.799/0.797/0.798；平：adult、bank、phoneme、miniboone、electricity；负：amazon 0.851/0.850/**0.867**、click 0.690/0.694/0.694 | 高基数类别（amazon）默认 min_data_per_group=100 太严 |
| M6 | amazon：min_data_per_group 1 / depth 5 / lambda 5 | 0.8134→0.8614→0.8683 | 与 XGBoost 0.8673 持平；airline 上 mdpg=1 反而降到 0.7774（100 时 0.7855），参数依数据而定 |
| M7 | 叶堆（O(log L) 取最大增益叶） | airline 2048 叶模型不变（0.7855），用时下降 | 保留 |
| N2 | split_penalty：增益减 c·ln(候选切分数)（多重比较校正，自创） | adult/bank/click 升 0.003–0.013；amazon/credit/electricity/phoneme 降 0.007–0.03；higgs 平 | 否决：非普遍收益，已回退 |
| M8 | 等预算调参 v2（三库共用同一组随机抽样映射到各自参数名，24 组） | 测试 AUC（mb/lgb/xgb）：adult .9265/.9253/.9259，amazon .8743/.8624/.8769，bank .9358/.9372/.9329，click .6940/.6929/.6968，credit .8479/.8449/.8449，electricity .9861/.9852/.9874，phoneme .9578/.9559/.9631，higgs .8005/.8019/.8021 | 小数据集测试集仅 1k–20k 行，AUC 标准误 0.003–0.005，差异多在噪声内；需多次重复划分才能下结论 |
| M9 | 同配置对照（用 XGBoost 选中的配置跑 moonboost） | phoneme .9613 vs .9631；higgs .8012 vs .8021；electricity .9865 vs .9870 | 纯数值数据上与 XGBoost 同配置基本持平（略低 0.001–0.002，噪声级） |
