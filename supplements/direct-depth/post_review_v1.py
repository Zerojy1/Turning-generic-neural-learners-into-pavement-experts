"""Post-computation documentation only: preserves frozen code, results and figures."""
from common import *
import platform, subprocess, shutil, xgboost

def table(df):
    cols=list(df)
    rows=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for _,r in df.iterrows():
        rows.append('| '+' | '.join(f'{v:.5f}' if isinstance(v,(float,np.floating)) else str(v) for v in r)+' |')
    return '\n'.join(rows)

def main():
    freeze_check()
    assert json.loads((ROOT/'checks/independent_supplement_verification.json').read_text())['passed']
    expected={'01_three_level_comparison.png':'511bb0aeaa982c5c1c016825b2b9712018677588ed3e8e9fdb184218439b9822',
              '02_calendar_rollout.png':'941dc66aed241fd8be7fd7769ce07be75f9077902e9e5ba3b71f2bf303c30f8b'}
    for name,sha in expected.items():assert digest(ROOT/'figures'/name)==sha
    original=ROOT/'checks/report_before_scientific_review.md'
    assert not original.exists(), 'Review already applied; inspect before retrying'
    shutil.copyfile(ROOT/'REPORT_ZH.md',original)
    t=ROOT/'results/tables'
    main=pd.read_csv(t/'three_level_main_table.csv');one=pd.read_csv(t/'one_step_summary.csv')
    sc=pd.read_csv(t/'scenario_summary.csv');iv=pd.read_csv(t/'interval_summary.csv')
    scenario=sc[(sc.scope=='past_common')&(sc.horizon==4)&(sc.scenario.isin(['realized','calendar_past','recent_temperature']))]
    pivot=scenario.pivot(index=['arm','family'],columns='scenario',values='macro_RMSE_mm').reset_index()
    missing=sc[(sc.scope=='full')&(sc.scenario=='missing_one')][['arm','family','n','macro_RMSE_mm','skill']]
    late=pd.read_csv(t/'late_load_summary.csv')[['arm','family','n','macro_RMSE_mm','skill']]
    selected=one[one.scope=='primary2089'][['arm','family','macro_RMSE_mm','macro_MAE_mm','skill','increment_R2','wins_vs_persistence']]
    int4=iv[(iv.scope=='past_common')&(iv.scenario=='calendar_past')&(iv.horizon==4)][['arm','family','mean_structure_coverage','worst_structure_coverage','MPIW_mm','MIS95_mm']]
    note='''## 核验后的定量解释与不利结果

### 三层对照回答了什么

在本数据、结构留出划分、既定网络容量和三档学习率预算下，完整EKI相对直接深度预测的结构均衡RMSE下降49.49%–57.31%，四个新增比较的Holm调整p为0.03720–0.03960，均达到预设阈值。每种骨干在15个测试结构中的14个上误差更低；不是逐结构全部胜出。

直接预测A0的技能为−0.865至−1.756，均弱于persistence。这个现象说明显式观测参考在本配置下提供了有用的归纳偏置，不能推出所有直接预测模型天然无效，或这些骨干经过其他合理设计也达不到更好精度。本轮A0与A2的输入、网络及初始化规则相同，区别是输出偏置/目标参数化；各自内层选择学习率与训练轮数，最终独立重拟合。相同搜索预算不等于已找到每个方法的全局最优。大幅百分比收益必须与A0绝对误差及persistence一起报告。

A2与A1的RMSE分别为0.301–0.343 mm与0.285–0.294 mm；原先4.03%–14.36%的降低仍仅指A1对A2。由A0到A2的RMSE差大于由A2到A1的差，支持把观测参考写成关键组成部分；这种顺序比较不等于将非线性组件的因果贡献精确分摊。相对树模型0.28947 mm，EKI-MLP/Transformer点估计略好，GRU/LSTM略差，本轮不新增“胜过XGBoost”的统计主张。

技能定义是先对结构MSE等权平均，再计算1−模型MSE/对应persistence MSE；RMSE为该平均MSE的平方根。它不是逐结构RMSE的算术平均，也不是逐结构技能的算术平均。区间为10000次配对结构bootstrap百分位区间；p值为预先冻结的边界居中bootstrap参考值，不是精确随机化检验。15个结构共享场地且训练集重叠，推断范围限于该协议。

'''+table(selected)+'''

### 多步递推：温度情景必须在相同目标集合比较

下表为严格历史同期可用的共同集合，四步均为1624个目标、15个结构，未来荷载在所有情景中相同且给定。列realized、calendar_past、recent_temperature分别表示实际未来温度、仅用起点前历史同期温度、近期温度延续，单位为mm。历史同期估计不使用未来年份；LOYO使用其他年份的版本只作回顾性敏感性，不能称为部署预测。

'''+table(pivot)+'''

历史同期四步EKI为0.424–0.449 mm，低于同骨干A2的0.468–0.545 mm和A0的1.030–1.255 mm；但XGBoost为0.40363 mm，仍优于四个EKI点估计。相同共同集合下，EKI使用历史同期温度较近期温度延续改善四个骨干，但仍弱于实际温度情景。这是对部署温度估计的描述性支持，不意味着各年气候不变、气候态无误差或所有温度组件被单独验证有效。

### 缺测：EKI的优势不能概括为所有场景都更稳

删除一次观测并重新聚合暴露后，以下2059目标结果显示：EKI仍优于直接预测，但四个骨干均弱于各自普通残差A2，且弱于XGBoost。EKI的RMSE约0.434–0.436 mm，对比A2的0.401–0.429 mm与XGBoost的0.39603 mm。该不利结果应在正文保留，不能写成“网络缺测更稳”的统一结论。

'''+table(missing)+'''

### 后段荷载与敏感性

累计荷载75%以后的494个目标是回顾性分层。EKI点估计在四个骨干上优于A0/A2和本集合的XGBoost，但没有预设独立显著性门槛；不能把这一分层解释为模型已在未来荷载外推中验证成功。2164全部目标的完整结果见one_step_summary.csv，三层误差排序与2089主集合一致。

'''+table(late)+'''

### 区间：平均覆盖率不能替代最差结构检查

一步EKI平均结构覆盖率94.96%–95.30%，MPIW为1.201–1.273 mm；直接预测的覆盖率84.39%–90.46%，且区间更宽（2.009–2.301 mm）。这些是经验校准结果，不是原生概率头识别了测量误差。

以下为历史同期共同集合四步区间。EKI平均覆盖率92.28%–94.56%，最差结构只有54.63%–78.70%，低于名义95%；尽管宽度和MIS相较A0/A2有所改善，不能声称递推区间已可靠覆盖所有结构。缺测时EKI的MIS约2.396–2.445，高于相应A2（2.278–2.394）及树模型2.217，进一步说明缺测收益受限。校准和模型选择共享内层验证结构，没有分布无关覆盖保证。

'''+table(int4)+'''

### 图表人工核对与论文表述

图01已经实际查看，12根柱值与三层主表一致，坐标为结构均衡RMSE且从0起始，图例和数字无遮挡。它展示点估计，配对置信区间放在主表，不应把没有误差条理解为没有不确定性。

图02已经实际查看，展示历史同期共同集合1–4步的三层预测误差；各面板纵轴量程不同，应比较每个面板内的数值，不能按线段视觉斜率比较骨干。横轴为预测步数，不是固定天数。该图不包含树模型，正文和表格必须同时报告树模型更优的四步结果。

正文可写：“在冻结的结构留出协议与既定训练预算下，工程知识嵌入改善了四类通用神经骨干相对于直接深度预测的精度，并在普通观测残差预测之上提供额外收益。收益随监测场景而变化，一次缺测及最差结构区间覆盖仍存在局限。”

原P1通过、P2未通过保持不变。新增A1对A0比较不弥补P2未建立非劣效的证据缺口，也不证明温度、荷载、结构各组件均贡献正收益。没有新增外部数据，没有证据证明达到人类专家水平或可以替代检测设备。本轮不改变原跨类别和归因稳定性结果。

'''
    old=original.read_text(encoding='utf-8')
    (ROOT/'REPORT_ZH.md').write_text(old.replace('## 建议论文安排',note+'## 建议论文安排'),encoding='utf-8')
    (ROOT/'REPRODUCTION_SUPPLEMENT.md').write_text('''# Direct-depth supplement: reproduction

This supplement adds 100 matched A0 models to the unchanged EKI distribution v1.0.0. It contains the actual frozen source, model weights (including inner candidates), numerical data, original reused predictions, learning curves, all scenario predictions and calibration records. The full original EKI weights remain in the base v1.0.0 release. The original P1 passed and P2 failed; both remain unchanged.

1. Verify the release ZIP against SHA256SUMS.txt. Check every member against reproduction_manifest.json (SHA-256 and byte counts); retain the manifest.
2. Extract to a new directory. Python packages and GPU context are listed in delivery_environment.json. Paths in runtime code resolve relative to that directory; provenance records retain the original machine's history.
3. Without refitting, run `python verify_supplement.py` on a CUDA-enabled environment to replay all 100 new models and recompute the primary metrics, paired bootstrap, Holm adjustment and 360 calibration groups. This writes verification logs in the extracted copy.
4. `python supplement_analysis.py` recomputes the three-level primary and sensitivity tables from saved predictions. `python supplement_evaluation.py` aggregates the saved scenario groups; inference results and inner calibration predictions are retained under evaluation/.
5. For a clean full refit, keep the original extraction intact and use a separate copy containing frozen root code/configuration, data/, base_reference/ and provenance checks. Omit runs/, evaluation/, results/ and queue state in that new copy, then run `python queue_train.py main`. This is optional and computationally expensive; it is not needed to verify the delivered results. Do not use outer performance to modify any settings.

The source freeze is 239199ecf24890a41ef481301827ac6f226aea46. training_freeze.json and analysis_freeze.json are byte-level guards. post_review_v1.py only enriches documentation after computation; it does not change numerical analysis or figures. checks/documentation_change_log.json records this distinction.

The new comparisons are further frozen exploratory comparisons on previously examined single-site data. A1 vs A0 measures the entire formulation, A2 vs A0 bridges the observation reference, and A1 vs A2 measures the additional EKI design. None establishes independent external confirmation.
''',encoding='utf-8')
    with (ROOT/'RESULT_INDEX.md').open('a',encoding='utf-8') as f:
        f.write('\n- REPRODUCTION_SUPPLEMENT.md：补充实验复现步骤\n- delivery_environment.json：运行环境\n- checks/visual_scientific_review.json：实际看图及科学解释核验\n- checks/documentation_change_log.json：仅文档补充记录\n')
    write(ROOT/'delivery_environment.json',{'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'xgboost':xgboost.__version__})
    write(ROOT/'checks/visual_scientific_review.json',{'passed':True,'time':stamp(),'reviewer':'assistant actual image inspection and table review','images':expected,'checked':['labels and unoccluded content','12 bar values match CSV','four rollout panels show recorded scope','distinct panel y ranges disclosed','direct baseline weaker than persistence disclosed','missing observation disadvantage disclosed','tree better in calendar four-step disclosed','worst-structure undercoverage disclosed','original P1 passed and P2 failed retained','exploratory and no external validation retained'],'freeze_check_passed':True})
    write(ROOT/'checks/documentation_change_log.json',{'version':'post_review_v1','time':stamp(),'script_sha256':digest(ROOT/'post_review_v1.py'),'training_freeze_unchanged':True,'numerical_analysis_unchanged':True,'figures_unchanged':True,'preserved_auto_report':str(original.relative_to(ROOT)),'changes':'Added table-grounded Chinese scientific interpretation, adverse results, figure reading notes and reproduction instructions. No fits or numerical results changed.'})
    freeze_check()
    print('DOCUMENTATION_REVIEW_COMPLETE',stamp())

if __name__=='__main__':main()
