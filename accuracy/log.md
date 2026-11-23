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
- 11-21 大基准（10 万行，8 万训练，4 种子，mdil 20，逐行配对 se）：abs .67929 vs XGB .68250 / LGB .67902；q0.1 .41051/.41452/.41021；q0.9 .41214/.41581/.41162。对 XGB 显著领先（5–10 se），对 LGB 落后约 0.0004（约 1 se/种子，12 次中 10 次为正）。小基准“abs 三方最佳”不成立，README 已改为大基准结果。
- 否决：lambda 0；lossguide 63 叶（对齐 LGB）；平滑 pinball 梯度（sigmoid，δ=0.1/0.3×平均|残差|，均更差，δ 越大越差）；LightGBM 分位数插值公式（dL 0.00036→0.00034，无差别）。quantile 与 LGB 的差距来源未定位。
- 11-21 排序目标（main f00dd9d）：rank:ndcg（LambdaRank，LightGBM 式：截断 30、ΔNDCG/(0.01+|Δs|)、log2(1+s)/s 归一）与 rank:pairwise（RankNet）；group 参数、CLI --group/--rank-truncation；XGBoost 加载 2 万行逐位一致。MSLR-WEB10K fold1，单线程 100 轮 eta0.1 255 叶 mcw 0.001：mb 40.0–41.3 s / NDCG@10 .5051；LGB 42.9 s / .5042；XGB 60.5 s / .5042。差 +0.0010±0.0019（不显著）。全对 52.3 s / .5020。
- 梯度优化：exp 每文档一次+一次除法+分支消除+Int64 键排序，梯度 33→18 s（2 叶 100 轮基准）；剩余每对约 3.5 ns。
- 分箱按 16 列条带转置：MSLR 3 轮 4.7→3.2 s，模型逐字节相同。
- 11-22 参数 JSON（main 1f9b81d，CI 绿）：TrainParams::from_json（XGBoost 名称+别名，数字可为字符串）；web train_with、webgc train_run_params（train_wasm.mjs 第 4 参数可传对象，非 4 种基础目标自动走对象）、CLI --params。浏览器端此前只能用 4 种目标与 5 个参数，现可用全部 15 目标/类别/采样/lossguide/排序。JS、wasm-gc、CLI 同参数模型逐字节一致。
- 11-22 rank:pairwise MSLR（单线程 100 轮 255 叶）：mb 38.9–39.4 s / .4915；XGB 默认 pairwise 53.6 s / .4923（差 −.0008±.0023）；XGB topk30 54.1 s / .4875（差 +.0040±.0022）。均不显著；pairwise 截断 0：41.3 s / .4871（更差）。
- 11-22 否决：窄直方图按 2/4/8 字（16/32/64 特征）分遍扫行，MSLR 30 轮 14.9 s → 18.3–22.7 s（更慢；行索引/梯度重复读取得不偿失）。
- 11-22 CSV 读取（main 38d6b1d，CI 绿，native 77/77）：原地扫描+精确小数快速路径（≤15 位有效数字、|exp|≤22 时 m×10^e 一次舍入，其余回落 parse_double，3000+ 随机串逐位相同）；from_csv_bytes 直接读 UTF-8。MSLR 579 MB 加载+分箱+1 轮：9.6 → 4.8 s；1M×8 GBM-perf 0.62 → 0.12 s（参考：pandas read_csv 读 MSLR 0.78 s，仍更快）。模型逐字节相同。
- 11-22 rank_gradients 预分配 FixedArray+unsafe（−3%）；按标签类别跳过同标签对（求和顺序不变，模型逐字节同；2 叶 100 轮 16.3→15.2 s，255 叶全对 30 轮 18.7→16.9 s），main 91646d3。
- 11-22 排序键专用快排替代泛型 sort（−3~4%，3ea3361）。单字节分箱按字整字写入（MSLR 建数据 2.84→2.16 s，GBM-perf 不变，81b699d）。
- 11-22 MSLR 全量（同会话）：mb 38.4 s/.5051，XGB 59.4/.5042，LGB 42.0/.5042；全对 52.3→46.0 s；pairwise 39.3 s 不变。参数 JSON 模型（binary/lossguide+subsample/quantile）XGBoost 加载预测逐位一致。
- 否决：量化直方图改 Int32（当前量化级 2^31/n，每行值约 12 位，节点和放不进 32 位打包）。
- 11-22 否决：binary:logistic 梯度单独成环（airline 255 叶 100 轮 5.32→5.27 s，噪声内）。部分热点：airline 255 叶 histogram 37%、partition 23%、gradients 11%、partition_copy 9%。
- 11-22 CLI 修正：--params 会覆盖其前面的选项（如 --group），改为在已有参数上叠加；TrainParams::from_json 加 base 参数（45e2869）。rank:ndcg/rank:pairwise 三组参数模型（63 叶 lossguide+subsample+lambda、depth6+colsample、全对 depth8+gamma）XGBoost 加载预测逐位一致。
- 11-22 否决：8 特征直方图两份交替累加（GBM-perf 100 轮 3.92→4.06 s，+3%，链式依赖不是瓶颈）。
- 11-22 否决：compact_at 改 1/3/4（MSLR 30 轮 14.1–14.4 s → 14.6–15.9 s，2 最优）。
- 11-22 否决：复制层加列存字节副本供分区读（逐行写 +25%；按列第二遍写入后 rank +1.5%、airline 255 叶 +21%）。perf：MSLR 直方图 41%，热点为行字载入的缓存缺失与直方图 RMW，偏访存受限。
- 11-22 分箱：四趟基数排序的字节计数一次扫完、16 列条带拷贝缓冲放局部变量（MSLR 1 轮总时间 4.09→3.97 s，GBM-perf 不变，模型逐字节同，af0163b）。perf：建数据剩余热点为基数排序散写与二分查找。
- 11-22 新目标 binary:hinge（378347e）：GBM-perf 80 万训练/20 万测试，100 轮 eta0.1：depth6 mb 2.2 s/.8144 vs XGB 4.7 s/.8145；depth10 3.2 s/.8237 vs 6.7 s/.8229；XGBoost 加载逐位一致；截距同 XGBoost（均值>0.5 取 1）。
- 11-22 新目标 survival:cox（beb6776）：合成 25 万行 38% 删失，200 轮 eta0.05：depth4 mb 1.3 s/C .7515 vs XGB 1.8 s/.7518；depth6 1.6 s/.7565 vs 2.5 s/.7559；逐位一致。exp 只算一次（2.2→1.6 s）。共 17 种目标。
- 11-23 rank:map（a580815）：XGBoost DeltaMAP 加权的 LambdaRank，0/1 标签。MSLR（相关度≥1 记为相关）100 轮 255 叶两次：mb 37.5–38.3 s MAP .6443，XGB 55.0–56.6 s .6419，LGB lambdarank 38.9–39.1 s .6427；对 XGB +0.0024±0.0009（显著），对 LGB +0.0016±0.0011（不显著）；XGBoost 逐位一致。
- 11-23 survival:aft（f08bf28）：区间删失，normal/logistic/extreme，XGBoost 梯度/截断/极限；erf 用 fdlibm 有理式、aft_exp=train_exp(x/16)^16 保证各后端同模型。合成 25 万行（50% 观测/25% 右/10% 左/15% 区间）200 轮 eta0.05：速度 1.4–2.4× 于 XGB，逐位一致。
- 11-23 假设：AFT 从最大似然常数起步而非 XGBoost 固定 0.5。4 个数据种子 depth4（逐行 nll 配对）：extreme 改善 0.013–0.024（稳定）、normal 改善 0.002–0.005、logistic 变差 0.003–0.007（4/4）。净收益为正，采用；logistic 变差原因未明（截距是精确 MLE）。extreme 截距与 scipy MLE 不同（1.72 vs 2.18）是因梯度截断 ±15，训练目标一致，不改。
- 11-23 分位叶值改用选择算法（9586c9f）：perf 显示单分位训练 51% 时间在每叶排序；改为 quickselect 取两个次序统计量（预算耗尽退回排序），模型逐字节不变。8 万行 100 轮 depth6 单线程：0.74–0.77 s → 0.33–0.39 s；对比 XGB 0.74–0.76、LGB 0.44–0.51。采纳。
- 11-23 多分位（0714e77）：quantile_alpha 列表 → num_target=K、每轮 K 棵树、base_score 向量；XGBoost 的 PredTransform 对每行预测排序（防交叉），初版未排序导致 7/20000 行不一致，补上后逐位一致；alpha 须升序（XGBoost 同样拒绝乱序）。4 种子 [0.1,0.5,0.9]：mb 1.07 s vs XGB 2.10 vs LGB（三个模型）1.44；pinball 全部优于 XGB；min_data_in_leaf 20 时对 LGB 12 项中 9 项略高、均约 1 se 内（不算领先）。多输出模型与逐个单分位模型的 margin 逐位相同（测试覆盖）。
- 11-23 多输出（num_target，main 于 0714e77 后一提交，CI 绿）：二维标签（行优先），squared/pseudohuber/absolute/logistic/logitraw；每输出用自己列的截距，每轮每输出一棵树 = XGBoost one_output_per_tree；测试证明每输出与单独训练的模型逐位相同。25 万行 10 特征 3 目标 3 种子：squarederror mb 1.88–1.95 s vs XGB 3.53–3.68 vs LGB(3 模型) 3.08–3.13；logistic 2.19–2.22 vs 4.12–4.30 vs 3.75–3.87；逐位一致；损失对 XGB 18 项中 10 项更低（≤3 se，持平），对 LGB 13/18 更低。注意：XGBoost 的 multi_output_tree（向量叶）未实现。
- 11-23 末层分裂与叶值合并为一遍（split_to_leaves：深度上限的两个子叶直接按行累加精确梯度并写 leaf_at，免分区+二次读）：GBM-perf 100 轮 5 次交替 base 3784–3955 vs new 3793–3838 ms，模型逐字节相同，差在噪声内 → 否决。
- 11-23 向量叶树 multi_output_tree（e132294，multi_tree.mbt）：单树多输出，增益=各输出增益之和；初版 Double 直方图 2.9–3.5 s（慢于 XGB 2.7）→ 每输出打包整数 (g<<32)+h 直方图、k=2/3 专用循环 2.4 s → 叶值改为建树后按行顺序精确求和+leaf_at（原先叶内随机 gather 占 grow 的 28%）1.8 s → 梯度去元组分配、量化无分支 1.3–1.7 s。3 种子：sq mb 1.37–1.42 vs XGB 2.71–2.85 vs LGB(3 模型) 3.45–3.47；logistic 1.66–1.71 vs 3.33–3.36 vs 4.09–4.16；XGBoost 加载逐位一致（18/18 True）；损失对 XGB 持平（低 7/18，|d|≤约 3 se），向量叶整体劣于每输出一棵树，LGB 三模型 18/18 更好（如实写入 README）。加载器把 size_leaf_vector 树展开为每输出一棵；XGBoost 生成的 fixture 逐位测试。四后端 90/90。主路径模型逐字节不变（GBM-perf、多输出）。
- 11-23 logistic 梯度：train_exp 手工内联进专用循环（标量）：Xh 100 轮 6 次交替 base 3069–3135 vs new 3075–3195 ms，模型同 → 否决（噪声，第二次确认标量专用环无益）
- 11-23 logistic 梯度用 core/v128 f64x2 两行一组（exp、sigmoid、float32 舍入、hessian 下限向量化；量化与求和仍标量）：native Xh：base 3027–3172 vs new 2880–3050 ms（约 −4%）；wasm-gc GBM-perf：6.01–6.09 vs 5.70–5.73 s（−5.5%）；模型逐字节同 → 合入 d6c2bf0；量化也向量化版反而慢（x86 无 trunc_sat_u 快速路径，回退 MoonBit 实现）
- 11-23 softmax 概率 f64x2 两行一组 + 缓冲跨轮复用：500k×10×3 类 50 轮：2525–2543 vs 2356–2380 ms（−6.5%），模型同 → 合入 fc0c113
- 11-23 GBM-perf 同会话复测（depth10 100 轮）：mb native 3.72/3.78 s，wasm-gc 5.63–5.82，JS 12.0；XGB 7.92/7.80；LGB 6.24/6.21；AUC .7537/.7533/.7545 → README 已更新
- 11-23 窄分箱数值分裂分区改为算术比较（bin=byte+add，免 left_table 查表）：GBM-perf native 3592–3656 → 4178–4239 ms（+15%），wasm-gc 5.64 → 6.19 s，模型同 → 否决（查表更快，算术链更长）
- 11-23 rank:ndcg 成对梯度 f64x2 两对一组（按类列表的 j）：MSLR 255 叶 30 轮 base 13328–13342 vs new 13034–13379 ms，模型同 → 否决（噪声；成对循环受 gather/scatter 限制）
- 11-23 窄分箱复制层分区改为按原行查列存 by8（orig 间接）：airline 类别 255 叶 6.3–7.1 → 8.2–8.9 s、Xh depth10 4.9–5.1 → 6.3–6.4 s，模型同 → 否决（1MB 列随机 gather 比顺序读 16B 行更慢）
- 11-23 lossguide 多级复制（recopy：首次复制下每 3 层把节点行的分箱/梯度/位置再复制成连续区，最多 6 级；更深层 orig 存一级位置，finish_leaf 经 orig 写 leaf_at）：airline 类别 255/1023/4095 叶 −15%/−20%/−25%（6.9–7.2→5.9–6.1，9.5–10.3→7.9，17.2–17.6→13.0），模型逐字节同，native/wasm-gc/JS CLI 输出同。调参：每 4 层次之、每 2 层与 3 层持平、加最小行数阈值（16k/64k）反而慢；depthwise（GBM-perf depth10）每 3/4 层复制为 +5%/持平 → 只对 lossguide 开启。合入 cbf03b0；README 同会话：mb 6.0/8.0/13.1 s vs LGB 6.5/9.4/19.9 vs XGB 21.2/34.3/58.7，AUC 不变
- 11-23 MSLR rank:ndcg 255 叶 30 轮（lossguide）复测 recopy：mbgc 13.26–13.49 s vs cbf03b0 12.51–12.65 s（−6%），模型同
- 11-23 行拷贝 words 循环展开为 copy_row（首两词显式）：airline 类别 1023 叶 6 次 −2%（6/6 更快），GBM-perf depth10 +0.6%（3/3 更慢），模型同 → 否决（净收益不明确）
- 11-23 注意：ibx 单次 exec 超时会杀掉其中 & 启动的后台任务，长任务须 setsid nohup … < /dev/null &
