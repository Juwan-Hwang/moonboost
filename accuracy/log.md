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
| M10 | airline 等叶数三库对比（oh32，100 轮） | 255 叶 mb .7631 / lgb .7612 / xgb .7547；1023 叶 .7792/.7761/.7638；4095 叶 .7883/.7829/— | 同树规模下 moonboost 每档都领先 |
| M11 | 类别只留 max_bin-1 个高频类（便于一字节 bin） | 1023 叶 .7792→.7779，4095 叶 .7883→.7868 | 小幅下降（噪声级），换来窄 bin 路径与统一语义；保留 |
| S1 | 速度剖析 lossguide+类别 1023 叶：histogram 43%、partition 23%；类别树更深更偏（加权平均叶深 12.4 vs 10） | 无类别 lossguide 8.7 s，有类别 12.5–14 s | 偏斜切分让 partition 成本上升 |
| S2 | lossguide 叶行号留在分区写入的缓冲（省掉每次切分后的整段回拷） | 4 次交替：11.6 s → 10.9 s（-6%），模型逐字节相同 | 保留 |
| S3 | 原生 histogram 为两字（9–16 特征）展开 | lossguide 255 叶 8.56→7.54 s（-12%）；depthwise 10 层 6.34→5.73 s（-10%）；模型相同 | 保留 |
| S4 | 偏斜切分（一侧 <10% hessian）用分支单写 partition | 1023 叶 9.28→9.73 s；depth10 5.34→5.48 s | 否决：分支版更慢，已回退 |
| T1 | 干净计时（单核顺序，airline+hour/minute，oh32） | 255 叶 mb 7.7 s/.7629，lgb 6.4/.7612，xgb 19.1/.7547；1023 叶 10.3/.7779，9.0/.7761，30.1/.7638；4095 叶 19.4/.7868，19.4/.7829 | 准确率每档领先；速度 255–1023 叶慢 LightGBM ~15%（S3 前测），4095 持平 |
| S5 | lossguide：分区时顺带累加（按 hessian 猜的）小侧直方图，省一次 gather（两字布局） | 255 叶 7.17/7.08 s → 9.08/8.14 s | 否决：分支阻碍向量化/预测失败，更慢 |
| C1 | 定位 amazon 落后：b5f99ff 起每个类别列只留 255 个高频类，amazon 第 0/1 列只覆盖 45%/29% 的行（其余并入缺失）；kdd_upselling 多列同样 | airline 各类别列覆盖 ≥99.8% | 原因确认，见 C2 |
| C2 | 高频类覆盖 <99% 行时改为按槽位上限保留全部类别（否则仍 255，保一字节路径） | airline 255 叶模型逐字节相同、用时不变；amazon 5 次划分均值 .8487→**.8592**（xgb .8607，lgb .8563）；XGBoost 读模型逐位一致 | 保留，044f5c9 进 main |
| R0 | 重复划分基线（rep.py：train+test 合并，5 次随机 80/20，每次每库 12 组等预算随机调参，按内层验证集选） | 测试 AUC 均值 mb/lgb/xgb：adult .9286/.9280/.9277，amazon .8487/.8563/.8607，bank .9348/.9355/.9345，click .6958/.6981/.6956，credit .7689/.7715/.7657，electricity .9804/.9789/.9811，higgs .8060/.8064/.8067，miniboone .9848/.9847/.9847，phoneme .9526/.9530/.9530 | 除 amazon 外全部在 ±0.003 内，即持平；没有一个数据集明显领先 |
| C3 | amazon 同配置对照（depth 6，3 次划分） | cat_smooth 0：mb .8514 vs xgb .8711；cat_smooth 1/10 + mdpg 0 + cat_l2 0：.8709/.8713 | 差距来自类别正则默认值，算法本身与 XGBoost 持平。XGBoost 按 -g/h 排序、并列时靠浮点误差随机打破；我们整数量化后并列是精确的，cat_smooth>0 让大类别优先 |
| S6 | 剖析 lossguide 255 叶：histogram 38%、partition 22%、主循环 8.5%、partition_copy 5.7% | — | 已有父减子、压缩拷贝；无明显低垂果实 |
| F1 | 新功能 subsample / colsample_bytree / colsample_bynode / seed（未采样行梯度置零，同 XGBoost） | depth 8，3 次划分：credit .7671→.7827（bynode .6）/.7847（ss .8）；bank .9352→.9364；默认参数模型逐字节不变 | 保留，ea97a65 进 main；调参空间 v2（space2.py）三库同时加入行/列采样后重跑 |
| S7 | histogram 跳过最后一个不满的行字里的填充字节（airline 10 特征：每行 16 次累加里 6 次落在固定的备用槽，形成跨行的存储依赖链） | 争用下 cycles:u 47.5G→37.8G（-20%），模型逐字节相同；新增 3/8/10/15/16/20 特征 × depthwise/lossguide 的一字节/两字节一致性测试 | 候选，待空闲机器干净计时（1d23b8e 在 acc-exp） |
| M12 | cat_crossfit 移植到当前代码，在 amazon 上测（depth 6，cs10，mdpg0，3 次划分） | .8713 → .8688；cs1 .8689；放开 max_cat_threshold .8601 | 否决（airline、amazon 都无收益） |
| R2 | 重复划分 SPACE=2（三库都加 subsample/colsample_bynode，5 划分×12 组） | 配对差 mb−lgb / mb−xgb（±标准误）：amazon +.0053±.0012 / −.0032±.0031；electricity +.0047±.0015 / −.0010±.0006；phoneme +.0052±.0019 / +.0003±.0006；kdd +.0038±.0022 / +.0053±.0045；credit +.0047±.009 / +.0034±.011；其余 |差|≤.0007 | 对 LightGBM：3 个数据集显著领先（>2.5 se），其余持平；对 XGBoost：全部持平（无显著领先或落后）。不能写“遥遥领先” |
| R3 | SPACE=3（moonboost 再加 min_data_in_leaf，同 12 组预算） | 相对 SPACE=2：electricity −.0040±.0018，phoneme −.0049±.0021，其余持平 | 固定预算下搜索空间变大反而更差；min_data_in_leaf 保留为参数（LightGBM 对齐），不进默认搜索空间 |
| S7b | 填充字节跳过，空闲机干净交替计时（核 2，airline 无类别） | lossguide 255：6.61→5.75 s（−13%）；depth 8：5.03→4.60 s（−9%）；lossguide 1023：10.9→9.5 s（−13%）；模型逐字节相同 | 保留，1d23b8e 进 main（CI 绿） |
| F2 | 新功能 sampling_method=gradient_based（MVS：保留概率 min(1, sqrt(g²+λh²)/μ)，λ=(均值|g|/均值h)²，保留行 g、h 除以概率，量化尺度重算）。XGBoost 只在 GPU 上有 | 用 rep2 选出的配置，3 划分×2 种子，MVS−uniform：subsample 0.5 时 adult +.0030、electricity +.0053、kdd +.0075、bank +.0015、amazon +.0030、credit −.0030；0.8 时多持平 | 同采样率下 MVS 明显优于均匀采样；但相对不采样的调参结果只是持平（半数行、精度不掉），不是精度提升。90188c0 进 main。要换成速度收益需让未采样行不进直方图（待做） |
| T1 | airline 同叶数带类别（air_time.py，核 2，争用：核 0/1 跑 MVS） | 255 叶 mb 11.7 s vs lgb 7.4 s vs xgb 29.4 s；1023 叶 12.7 vs 10.7 vs 44.1；4095 mb 22.5 | 带类别时 255 叶慢 LightGBM 58%，比无类别时差得多，下一步剖析 |
| T2 | airline+类别同叶数，空闲机干净计时（含 S7） | 255 叶 mb 7.2 s / lgb 7.0；1023 叶 11.1 / 9.9 / xgb 36.5；4095 叶 19.7 / 20.3 / 64.6；T1 的 11.7 s 是争用造成的 | 255 叶基本持平，1023 叶仍慢 12%。剖析（带类别 255 叶）：histogram 38%、partition 27%、gradients 6%、partition_copy 5% |
| F3 | 新训练目标：reg:logistic、binary:logitraw、count:poisson（max_delta_step 0.7）、reg:gamma、reg:tweedie、reg:pseudohubererror、reg:squaredlogerror；梯度量化尺度改为算完梯度后实测 | 合成 16k 行，100 轮 depth 6：全部 XGBoost 加载后逐位一致；留出损失 mb/xgb：poisson .14831/.14629，gamma 1.26533/1.26800，tweedie 4.93330/4.93413，pseudohuber .98422/.99400，sqlog .18395/.18536，logistic .62110/.62099，logitraw .64138/.64120；默认目标模型逐字节不变 | 进 main（19e1a77），README 加表（bench/train_objectives.py） |

## 11-20 采样只在保留行上建树（bag 分支 → main 6ce81a3+）
- 假设：subsample/MVS 未保留行不进直方图/分区，树建完后把它们按原始分箱沿树分流加叶值，可提速且模型不变。
- 结果（专用机核 2，交替 3 次，airline Xh 无类别参数）：depth 8 sub0.5 4.39→3.03 s（−31%，SAME）；sub0.8 4.13→3.64（−12%，SAME）；lossguide255 sub0.5 5.07→3.27（−35%，SAME）；MVS lossguide255 7.83→6.60（−16%，SAME）；不采样 depth8/lossguide255 持平（SAME）。
- 首版分流用 by_col 且采样循环带分支只快 7–10%；改无分支采样 + 一字节 by8 分流后达上述数字。
- 语义变化：节点行数（min_data_in_leaf、类别 per_row）改按保留行计，与 LightGBM bagging 一致；带类别时模型与旧版不逐字节相同（airline 255 叶 AUC .7607→.7619，1023 叶 .7734→.7731，噪声内）。
- MVS 阈值改为两遍精确解（按 double 高 16 位分桶 + 桶内排序），替代 40 次二分：6.70→4.82 s（−28%），模型字节不同但 10 万测试行预测完全相同。
- 对手同参数（airline 带类别，subsample 0.5，100 轮）：255 叶 mb 4.0 s/.7619，LGB bagging0.5 11.3 s/.7596（不 bagging 6.4 s/.7612），XGB 22.2 s/.7471；1023 叶 mb 6.1/.7731，LGB 14.3/.7716（不 bagging 9.0/.7761），XGB 33.6/.7463。注意 LGB 不采样 1023 叶 AUC .7761 高于所有采样结果。
- 结论：保留，已合入 main。

## 11-21
- 剖析（airline 带类别 lossguide 1023 叶，main）：histogram 32%、partition 28%、best_split 6%、gradients 5%、partition_copy 4%、malloc/free ~5%。加权平均叶深 13.8；partition ≈1.9 ns/行·层，已近极限。
- H：原生端直方图缓冲按 slot 复用（同 JS），省分配。A/B 1023 叶 4 次：8.81/9.07/9.22 vs 8.74/9.26/9.20 s，噪声内。否决（未合入）。
- 重新干净计时 1023 叶（核 2，同一会话）：moonboost 8.8 s AUC .7779，LightGBM 9.2 s .7761，XGBoost 32.8 s .7638——11-20 填充字节与采样改动后 1023 叶已不再慢于 LightGBM（此前 12% 差距来自旧版本/争用）。
- 新目标 reg:absoluteerror、reg:quantileerror（7c85764）：按符号梯度长树，叶值改为叶内残差 alpha 分位数×eta（XGBoost Quantile 插值）；base_score 为标签分位数（与 XGBoost 3.4.1 写出的值一致）。XGBoost 加载逐位一致（3 种子×3 配置全 True）。
- 合成数据（t(2) 噪声、异方差，3 种子平均 pinball）：默认参数 abs .67621 vs XGB .68056 / LGB .67464；q0.1 .39413 vs .39019/.39078；q0.9 .42786 vs .42479/.42346。加 min_data_in_leaf 20（LGB 默认）：abs .67368（三方最佳），q0.1 .39308（最差），q0.9 .42441（介于两者）。
- 观察：XGBoost 3.4.1 的 quantileerror 叶值等于 base_weight，并非叶内残差分位数（原因未查）；我们把分位数叶换成 Newton 叶后更差（q0.1 .40041、q0.9 .43764 均值），故保留分位数叶。分位数目标不领先，如实记录。
- 11-21 4095 叶：moonboost 16.1 s .7868 vs LGB 19.8 s .7829 vs XGB 53.9 s .7655；255 叶本次 6.9 vs 6.5 s（另一次 6.4），持平。README 表已更新。
