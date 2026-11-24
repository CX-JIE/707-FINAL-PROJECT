"""生成中文查阅报告（自包含 HTML）。用法: python 06_report_zh.py <results_dir> <out_html>"""
import sys, os, json, base64, html
import pandas as pd

R, OUT = sys.argv[1], sys.argv[2]
J = lambda f: json.load(open(os.path.join(R, f)))
C = lambda f: pd.read_csv(os.path.join(R, f))


def img(f, w="100%"):
    b = base64.b64encode(open(os.path.join(R, f), "rb").read()).decode()
    return f'<img src="data:image/png;base64,{b}" style="max-width:{w}">'


def tab(df, fmt=3):
    return df.to_html(index=False, float_format=lambda x: f"{x:.{fmt}f}", border=0, classes="t", na_rep="—")


ri = J("run_info.json"); v = ri["versions"]
perf = C("performance_setA_cv.csv"); fv = C("fold_variability.csv")
boot = C("paired_bootstrap.csv"); ext = C("performance_external_sets.csv"); thr = C("threshold_metrics.csv")
cx = C("model_complexity.csv"); gc = J("gam_complexity.json"); ts = J("tree_size.json")
miss = C("core15_missingness_setA.csv"); vent = C("ventilation_vs_missingness_setA.csv")

ZH = {"primary": "主要比较", "supplementary": "补充比较", "exploratory": "探索性", "sensitivity": "敏感性分析", "primary (clinical)": "临床基准"}
order = ["GAM-7", "XGB-full", "SAPS-I", "SOFA", "XGB-7", "Tree-7", "LASSO-15", "Ridge-15", "XGB-15", "XGB-full (native NA)", "GAM-7 + NA indicators"]
p = perf.merge(fv[["model", "AUROC_sd"]], on="model").set_index("model").loc[order].reset_index()
p["group"] = p["group"].map(ZH)
p = p.rename(columns={"model": "模型", "group": "比较层级", "n_features": "特征数", "n": "样本数", "AUROC_sd": "AUROC 折间SD",
                      "cal_intercept": "校准截距", "cal_slope": "校准斜率"})

b = boot.copy()
b["ΔAUROC (95% CI)"] = b.apply(lambda r: f"{r.dAUROC:+.3f} ({r.dAUROC_lo:+.3f}, {r.dAUROC_hi:+.3f})", axis=1)
b["ΔBrier (95% CI)"] = b.apply(lambda r: f"{r.dBrier:+.4f} ({r.dBrier_lo:+.4f}, {r.dBrier_hi:+.4f})", axis=1)
b = b[["comparison", "n", "ΔAUROC (95% CI)", "ΔBrier (95% CI)"]].rename(columns={"comparison": "比较（前者 − 后者）", "n": "样本数"})

e = ext.pivot_table(index="model", columns="set", values="AUROC").reindex(["XGB-full", "GAM-7", "XGB-7", "LASSO-15", "Ridge-15", "Tree-7", "SAPS-I", "SOFA"]).reset_index()
e.columns = ["模型", "Set B AUROC", "Set C AUROC"]
cvA = perf.set_index("model")["AUROC"]
e.insert(1, "Set A 交叉验证 AUROC", e["模型"].map(cvA))

th = thr[thr.model.isin(["GAM-7", "XGB-full", "XGB-7", "LASSO-15", "Tree-7", "SAPS-I", "SOFA"])].copy()
th["threshold"] = th["threshold"].str.replace("training-fold prevalence (~0.14)", "训练折死亡率（约0.14）", regex=False)
th = th[["model", "threshold", "TP", "FP", "FN", "TN", "sens", "spec", "PPV", "NPV"]].rename(
    columns={"model": "模型", "threshold": "阈值", "sens": "敏感度", "spec": "特异度"})

cxz = cx.rename(columns={"model": "模型", "predictors": "预测变量", "size": "模型规模", "interpretability": "可解释形式"})

css = """body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",Segoe UI,Helvetica,Arial,sans-serif;max-width:1050px;margin:32px auto;padding:0 20px;color:#1f2933;line-height:1.75}
h1{font-size:25px;margin-bottom:4px}h2{border-bottom:2px solid #e4e7eb;padding-bottom:4px;margin-top:40px}h3{margin-top:24px}
.t{border-collapse:collapse;font-size:13px;margin:10px 0}.t th,.t td{padding:4px 9px;border-bottom:1px solid #e4e7eb;text-align:right}.t th{background:#f5f7fa}
.t td:first-child,.t th:first-child{text-align:left}.key{background:#effcf6;border-left:4px solid #27ab83;padding:12px 16px;margin:14px 0}
.warn{background:#fff8e6;border-left:4px solid #f0b429;padding:10px 14px;margin:14px 0}.meta{color:#616e7c;font-size:14px}code{background:#f5f7fa;padding:1px 4px}
.ck td:nth-child(3){text-align:center}"""

h = [f"<html><head><meta charset='utf-8'><title>707 期末项目 · 中文查阅报告</title><style>{css}</style></head><body>"]
h.append("<h1>简单可解释模型 vs 黑箱模型：预测 ICU 患者院内死亡</h1>")
h.append("<p class='meta'>707 Final Project · 中文查阅报告 · 数据：PhysioNet/CinC Challenge 2012 · 分析严格按照 Proposal v4（2026-10-15）执行 · 英文完整技术报告见 <code>report.html</code></p>")

h.append("""<h2>一、结论速览</h2><div class='key'>
<p><b>研究问题：</b>只用 ICU 入院后前 24 小时的数据，一个简单、可解释的模型能否达到与复杂机器学习模型<b>相当</b>的院内死亡预测性能？</p>
<p><b>回答：在区分能力上不能。</b></p>
<ul>
<li>XGBoost（全部 193 个特征）交叉验证 AUROC <b>0.826</b>，7 变量 GAM 为 <b>0.736</b>。配对 bootstrap 差值 <b>ΔAUROC = +0.091（95% CI 0.072–0.110）</b>，Brier 分数也是 XGBoost 更好（−0.012，CI −0.016 至 −0.009）。</li>
<li>在留出的 set B 和 set C 上结论不变（XGB-full 0.848 / 0.837；GAM-7 0.738 / 0.721）。决策曲线在 0.05–0.40 的所有阈值上都是 XGB-full 净获益最高。</li>
<li><b>差距的来源：</b>约 <b>30%</b> 来自模型灵活性（同样 7 个变量，XGB-7 比 GAM-7 高 0.026），约 <b>70%</b> 来自更多信息（XGB-full 比 XGB-7 高 0.064）。这些额外信息主要是<b>病情走势</b>（最后一次的 GCS、BUN）和<b>测量过程</b>本身（呼吸频率、pH、GCS 记录了多少次，总尿量）。</li>
<li><b>GAM 的价值：</b>它是可解释模型中表现最好的，高于 LASSO-15 和 Ridge-15（都是 0.720）以及决策树（0.689）。校准接近理想（截距 −0.001，斜率 0.97），比 SAPS-I（0.650）和 SOFA（0.623）高 0.09–0.11。它透明、校准良好、优于传统评分，但不能替代 XGBoost。</li>
</ul></div>""")

h.append("<h2>二、数据与队列</h2>")
h.append(f"""<ul>
<li><b>数据：</b>PhysioNet/CinC Challenge 2012 v1.0.0，许可 ODC-BY 1.0。引用：Silva et al., <i>Computing in Cardiology</i> 2012;39:245–248；Goldberger et al., <i>Circulation</i> 2000;101:e215–e220。</li>
<li><b>队列：</b>Set A 是开发集，共 4,000 例，没有排除任何患者。院内死亡 <b>{ri['deaths']}</b> 例（{ri['mortality_pct']:.1f}%），存活 {ri['n']-ri['deaths']} 例。Set B（死亡 568 / 存活 3,432）和 Set C（死亡 585 / 存活 3,415）只用作外部验证。</li>
<li><b>预测窗口：</b>只使用时间戳 ≤ 24:00 的测量值。生理上不可能的值（例如 HR ≤ 0、体重 = −1）设为缺失；百分比形式的 FiO₂ 换算为小数。</li>
<li><b>特征集：</b>小特征集 7 个（年龄、GCS 最小值、HR 均值、MAP 均值、呼吸频率均值、SaO₂ 最小值、BUN 最大值）；核心特征集 15 个；全特征集 193 个（37 个时序变量各取均值/最小/最大/最后一次/测量次数，加上静态变量和衍生变量）。</li>
</ul>""")
h.append("<h3>重要的数据发现：结构性缺失</h3>")
h.append(tab(vent.rename(columns={"mech_vent_24h": "24h内机械通气", "n": "人数", "RespRate_missing_pct": "呼吸频率缺失%", "SaO2_missing_pct": "SaO₂缺失%", "PaO2_missing_pct": "PaO₂缺失%"}), 1))
h.append("""<p>呼吸频率在机械通气患者中 <b>100%</b> 缺失（只对自主呼吸的患者记录），整体缺失 72.6%；SaO₂ 缺失 62.3%。这类缺失不是随机的，"有没有记录"本身就反映病情。按照 Proposal §3.2，<b>在看到任何结局之前</b>就定好了处理方式：主分析按方案使用训练折内的中位数填补；另做一个敏感性分析，加入"未记录/机械通气"指示变量。</p>""")

h.append("<h2>三、方法概要</h2>")
h.append(f"""<ul>
<li><b>Logistic GAM（7 个变量）：</b><code>pygam.LogisticGAM</code>，每个变量用惩罚三次样条，平滑参数 λ 由内层交叉验证选择（最终有效自由度 {gc['GAM-7_edof']:.1f}）。</li>
<li><b>浅层决策树（7 个变量）：</b>深度和最小叶子数由内层交叉验证选择，最终深度 {ts['depth']}、{ts['n_leaves']} 个叶子。</li>
<li><b>LASSO / Ridge 逻辑回归（15 个变量）：</b>L1 / L2 惩罚，C 由内层交叉验证选择。</li>
<li><b>XGBoost：</b>全部特征 / 7 个 / 15 个三个版本，内层 5 折随机搜索 25 组超参数；另加一个原生缺失值处理的敏感性分析。</li>
<li><b>SAPS-I / SOFA：</b>直接取官方结局文件中的评分作为临床基准。AUROC 用原始评分计算；概率由训练折内的逻辑回归重新校准得到。</li>
<li><b>验证：</b>分层 5 折嵌套交叉验证，所有模型共用同一套外层折（seed 707），所有填补和调参都只在训练折内进行。</li>
<li><b>评价指标：</b>AUROC、AUPRC、Brier、校准截距/斜率、十分位校准图、两个阈值下的混淆矩阵、决策曲线、各折波动，以及 2,000 次配对 bootstrap。</li>
<li><b>事后解释：</b>SHAP（TreeExplainer）、XGBoost gain 重要性、PDP/ICE、SHAP 交互值。</li>
</ul><p class='meta'>软件版本：xgboost {v['xgboost']} · pygam {v['pygam']} · shap {v['shap']} · scikit-learn {v['sklearn']}（均来自官方 conda-forge 源）</p>""")

h.append("<h2>四、主要结果</h2><h3>4.1 交叉验证性能（Set A，折外预测）</h3>" + tab(p, 3))
h.append(img("fig_roc_calibration.png"))
h.append("<h3>4.2 配对 bootstrap 差值</h3><p class='meta'>ΔAUROC &gt; 0 表示前者区分能力更好；ΔBrier &lt; 0 表示前者概率更准确。95% CI 不含 0 即视为差异明确。</p>" + tab(b))
h.append("<h3>4.3 外部验证（模型在全部 Set A 上重新拟合）</h3>" + tab(e, 3))
h.append("<p>外部集上的排名基本不变：XGB-full 在 B/C 上的 AUROC 甚至略高于交叉验证，说明没有过拟合。各模型在 Set C 上的校准截距都偏高（0.06–0.13），与 Set C 死亡率较高（14.6%）一致，实际使用时需要按当地基线死亡率重新校准。</p>")
h.append("<h3>4.4 阈值指标</h3>" + tab(th, 3))
h.append("<p>以训练折死亡率（约 0.14）作阈值时，XGB-full 比 GAM-7 多识别 35 例死亡，同时少报 228 例假阳性；在阈值 0.30 下，两者识别的死亡分别为 223 例和 132 例。</p>")
h.append("<h3>4.5 决策曲线</h3>" + img("fig_decision_curves.png", "70%"))
h.append("<p>阈值 0.15 时，净获益 XGB-full 为 0.062，GAM-7 为 0.046，相当于每 1,000 名患者多约 15 个真阳性，且不增加假阳性代价。阈值超过 0.30 后 GAM-7 的净获益趋近 0；SAPS-I 和 SOFA 在阈值 ≥ 0.20 时已接近「不干预」。</p>")
h.append("<h3>4.6 模型复杂度对比</h3>" + tab(cxz, 3))

h.append("<h2>五、可解释性</h2><h3>5.1 GAM 偏效应（95% CI）与 XGBoost SHAP 依赖曲线</h3>" + img("fig_partial_effects.png"))
h.append("""<ul>
<li><b>一致的地方：</b>年龄越大风险越高，近似线性（LASSO 中系数最大，0.47/SD）；BUN 越高风险越高（低到中等水平时上升最陡）；心率呈 U 型，约 80–85 次/分时风险最低，超过 100 次/分后明显上升；GCS 越高风险越低。决策树第一次分裂在 BUN ≤ 27.5，之后依次是 GCS、年龄和心率。</li>
<li><b>XGB-7 利用了填补值：</b>它的呼吸频率 SHAP 曲线恰好在中位数处出现尖峰。那些点是被中位数填补的患者，也就是机械通气患者，树模型把这个填补值当成了"在上呼吸机"的信号，而加性 GAM 做不到这一点。给 GAM 加上指示变量后 AUROC 提高 0.015（0.006–0.024），补回了这部分灵活性差距的约 57%。</li>
<li><b>GCS 最小值的问题：</b>GAM 中 GCS 曲线不单调（GCS 4–6 的风险反而高于 3），很可能因为镇静、插管患者的 GCS 被记为 3，混淆了这个变量的含义。XGB-full 更依赖"最后一次 GCS"，它反映的是恢复情况。</li>
<li><b>XGB-full 中这 7 个变量的曲线几乎是平的：</b>原因是信息被高度相关的变量（GCS_last、BUN_last 等）承担了。这是归因被分摊，不代表这些变量不重要。</li>
</ul>""")
h.append("<h3>5.2 XGB-full 的 SHAP 摘要</h3>" + img("fig_shap_summary_xgb_full.png", "70%"))
h.append("""<p>最重要的特征依次是：GCS_last、年龄、总尿量、BUN_last、呼吸频率测量次数、pH 测量次数。呼吸频率测得多（说明患者能自主呼吸）对应风险降低；pH 测得多也对应风险降低，可能反映心脏术后的常规血气检查（ICU 类型 2，死亡率低）；GCS 测得多（神经监测更密切）对应风险升高。这些"诊疗过程"特征能预测结局，但解释的是医护行为而不是生理状态，可能无法推广到其他医院。</p>""")
h.append(f"<h3>5.3 决策树规则（深度 {ts['depth']}，{ts['n_leaves']} 个叶子）</h3><pre style='font-size:12px;background:#f5f7fa;padding:10px'>{html.escape(open(os.path.join(R, 'tree_rules.txt')).read())}</pre><p class='meta'>weights 一栏为 [存活, 死亡] 人数。</p>")
h.append("""<h3>5.4 LASSO / Ridge 与交互作用</h3><p>LASSO 和 Ridge 的系数几乎相同，性能也无差别（ΔAUROC 0.000，CI −0.002 至 0.002）。MAP 和肌酐几乎被压缩到 0（MAP 在 5 折中只入选 2 次）；胆红素系数第四大（0.26/SD），但缺失率达 63%。XGB-7 中确实存在交互作用（最强的是 MAP×BUN、GCS×MAP、年龄×BUN），但主效应占每个变量 SHAP 贡献的 84% 以上，所以对这 7 个变量来说，加性结构已能捕捉大部分信号。</p>""")

h.append("""<h2>六、敏感性分析</h2><ul>
<li><b>XGB-full 原生缺失值处理 vs 中位数填补：</b>ΔAUROC = +0.001（−0.004 至 0.005），XGBoost 的结果不受填补方式影响。</li>
<li><b>GAM-7 + 缺失指示变量 vs GAM-7：</b>ΔAUROC = +0.015，ΔBrier = −0.0014，两者都显著。说明按方案预先定下的中位数填补略微低估了紧凑模型的潜力。</li>
</ul>""")

h.append("""<h2>七、结论</h2><ol>
<li>在本数据中黑箱模型明显更好（ΔAUROC 约 0.09），外部验证和决策曲线都一致。按照 Proposal §1.1 的"相当"标准，7 变量 GAM 不能替代 XGBoost。</li>
<li>性能提升约 70% 来自更多信息（病情走势、测量过程、更多变量），约 30% 来自模型灵活性；灵活性带来的提升中，约一半可以通过加入缺失/通气指示变量在可解释框架内补回。</li>
<li>在特别看重透明度的场景，GAM-7（或 GAM-7 + 缺失指示变量）是合理的选择：校准良好，AUROC 比 SAPS-I 和 SOFA 高 0.09–0.11，每一项都能画图说明。下一步值得尝试用更大的特征集构建可解释模型（例如加入 GCS_last、尿量、通气状态），检验能否在不用黑箱的情况下缩小信息差距。</li>
<li>事后解释显示，黑箱的部分优势依赖诊疗过程信号。这些信号有预测力，但不是因果，也可能无法跨机构推广，这本身就是部署黑箱模型前需要谨慎的理由。</li>
</ol>""")
h.append("""<h2>八、局限性</h2><ul>
<li>事后解释（SHAP/PDP）只描述模型怎样做出预测，不代表因果效应；变量相关时，归因会被分摊。</li>
<li>呼吸频率和 SaO₂ 存在结构性缺失，中位数填补掩盖了"未记录"这一信息。</li>
<li>XGBoost 选中的树数达到搜索上限 600，再增加树数性能可能略有提升，但这只会加强"XGB-full 领先"的结论。</li>
<li>数据来自单一来源，患者构成和测量频率限制了结论的推广；本研究不能证明临床获益。</li>
</ul>""")

ck = [("§2.5", "数据核查：结局编码、单位、记录数、死亡/存活人数、变量完整性", "✅", "英文报告 §1–2；本报告 二"),
      ("§3.1–3.3", "7 / 15 / 全特征三套特征集，只用前 24 小时", "✅", "code/featuresets.py；英文报告 §2"),
      ("§3.4", "重复测量的预设汇总方式（GCS 最小值、HR 均值、BUN 最大值等）", "✅", "code/01_build_features.py"),
      ("§3.5", "缺失比例/模式、分布、异常值、冗余；训练折内中位数填补", "✅", "英文报告 §2.1–2.4"),
      ("§3.5", "补充分析：XGBoost 原生缺失值处理", "✅", "本报告 六"),
      ("§3.6/4.4", "XGBoost 7 变量受控对照", "✅", "本报告 4.1–4.2"),
      ("§4.2", "Logistic GAM + 浅层决策树", "✅", "本报告 4.1、5.1、5.3"),
      ("§4.3/概览", "LASSO（及 Ridge）逻辑回归，内层交叉验证调参", "✅", "本报告 4.1、5.4"),
      ("§4.4", "XGBoost 全特征（及 15 变量），训练集内调参", "✅", "本报告 4.1"),
      ("§4.5", "SAPS-I、SOFA 临床基准", "✅", "本报告 4.1–4.3"),
      ("§4.6", "主要/补充/探索性比较，同一套外层折", "✅", "results/outer_folds_setA.csv"),
      ("§5.1", "分层 5 折嵌套交叉验证，预处理与调参都在训练折内", "✅", "code/03_models.py"),
      ("§5.2", "AUROC、AUPRC、校准图、Brier", "✅", "本报告 4.1"),
      ("§5.2", "阈值混淆矩阵：敏感度/特异度/PPV/NPV", "✅", "本报告 4.4"),
      ("§5.2", "决策曲线分析", "✅", "本报告 4.5"),
      ("§5.2", "各折波动、GAM 偏效应、LASSO 系数、复杂度对比、树深度与叶子数", "✅", "本报告 4.1、4.6、5"),
      ("§5.3", "配对 bootstrap ΔAUROC / ΔBrier 及 95% CI", "✅", "本报告 4.2"),
      ("§2.2", "Set B（及 C）作为独立测试集", "✅", "本报告 4.3"),
      ("§6.2", "特征重要性、SHAP、PDP", "✅", "本报告 5.2；英文报告 5.4–5.6"),
      ("§6.3", "XGBoost 与 GAM / 决策树 / LASSO 的解释对比", "✅", "本报告 五"),
      ("§6.4", "局限性", "✅", "本报告 八"),
      ("§7.1", "数据处理脚本、分析代码、数据来源/许可/引用说明；不提交大数据文件", "✅", "code/、run_all.sh、environment.yml；未做任何 git 提交")]
h.append("<h2>九、Proposal 完成情况对照表</h2>" + pd.DataFrame(ck, columns=["Proposal 条目", "要求", "状态", "对应位置"]).to_html(index=False, border=0, classes="t ck", escape=False))

h.append("""<h2>十、文件索引与复现</h2><ul>
<li><code>report.html</code>：英文完整技术报告（包含全部表格和图）；<code>report_zh.html</code>：本中文查阅报告</li>
<li><code>code/01_build_features.py</code> 提取 24 小时特征 → <code>02_data_quality.py</code> 数据质量 → <code>03_models.py</code> 主要建模 → <code>05_supplement.py</code> Ridge 与复杂度表 → <code>04_report.py</code> / <code>06_report_zh.py</code> 生成报告</li>
<li><code>results/</code>：全部结果表（CSV/JSON）和图；<code>data_derived/</code>：患者层面的特征表</li>
<li>复现：<code>conda create -n icu707 --override-channels -c https://conda.anaconda.org/conda-forge python=3.11 numpy pandas scipy scikit-learn matplotlib xgboost pygam shap</code>，然后运行 <code>conda run -n icu707 bash run_all.sh</code></li>
</ul>""")
h.append("</body></html>")
open(OUT, "w").write("\n".join(h))
print("wrote", OUT)
