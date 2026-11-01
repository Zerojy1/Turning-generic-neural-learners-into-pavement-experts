from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def mdtable(df):
 cols=list(df);lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
 for _,r in df.iterrows():lines.append('| '+' | '.join(f'{v:.5f}' if isinstance(v,(float,np.floating)) else str(v) for v in r)+' |')
 return '\n'.join(lines)

def main():
 freeze_check();assert json.loads((ROOT/'checks/independent_supplement_verification.json').read_text())['passed'];tables=ROOT/'results/tables';figs=ROOT/'figures';figs.mkdir(exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 main=pd.read_csv(tables/'three_level_main_table.csv');families=['mlp','gru','lstm','transformer'];labels=['MLP','GRU','LSTM','Transformer'];main=main.set_index('family').loc[families].reset_index();colors={'A0':'#8F99A6','A2':'#4384B5','A1':'#DA8744'}
 fig,ax=plt.subplots(figsize=(8.6,4.7),layout='constrained');x=np.arange(4)
 for i,(col,title,arm) in enumerate([('direct_RMSE_mm','Direct depth','A0'),('residual_RMSE_mm','Observation residual','A2'),('EKI_RMSE_mm','Full EKI','A1')]):
  bars=ax.bar(x+(i-1)*.24,main[col],.23,label=title,color=colors[arm]);ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
 ax.set_xticks(x,labels);ax.set_ylabel('Structure-balanced RMSE (mm)');ax.set_ylim(0,main[['direct_RMSE_mm','residual_RMSE_mm','EKI_RMSE_mm']].to_numpy().max()*1.22);ax.legend(loc='upper center',ncol=3,frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 for ext in ['png','pdf']:fig.savefig(figs/f'01_three_level_comparison.{ext}',dpi=220)
 plt.close(fig)
 scenario=pd.read_csv(tables/'scenario_summary.csv');fig,axes=plt.subplots(2,2,figsize=(9.4,6.7),sharex=True,layout='constrained')
 for f,label,ax in zip(families,labels,axes.ravel()):
  for arm,name in [('A0','Direct depth'),('A2','Observation residual'),('A1','Full EKI')]:
   z=scenario[(scenario.scope=='past_common')&(scenario.scenario=='calendar_past')&(scenario.family==f)&(scenario.arm==arm)].sort_values('horizon');ax.plot(z.horizon,z.macro_RMSE_mm,'o-',label=name,color=colors[arm],lw=1.8,ms=4)
  ax.set_title(label);ax.set_xticks([1,2,3,4]);ax.set_ylabel('RMSE (mm)');ax.set_xlabel('Forecast horizon');ax.grid(alpha=.18)
 axes[0,0].legend(frameon=False,fontsize=8)
 for ext in ['png','pdf']:fig.savefig(figs/f'02_calendar_rollout.{ext}',dpi=220)
 plt.close(fig)
 decision=json.loads((ROOT/'results/supplementary_decision.json').read_text());co=pd.read_csv(tables/'supplementary_primary_contrasts.csv');sums=pd.read_csv(tables/'one_step_summary.csv');late=pd.read_csv(tables/'late_load_summary.csv');interval=pd.read_csv(tables/'interval_summary.csv')
 speed=[]
 for s in specs():
  c=json.loads((ROOT/'runs'/s['job_id']/'complete.json').read_text());speed.append(dict(family=s['family'],seconds=c['seconds'],parameters=c['parameter_count'],peak_gpu_bytes=c['peak_gpu_bytes'],inference256_seconds=float(np.median(c['inference_256_seconds']))))
 pd.DataFrame(speed).groupby('family',as_index=False).agg(fits=('seconds','size'),median_job_seconds=('seconds','median'),parameters=('parameters','first'),max_gpu_bytes=('peak_gpu_bytes','max'),median_inference256_seconds=('inference256_seconds','median')).to_csv(tables/'new_compute_summary.csv',index=False)
 detail=[]
 for r in main.itertuples():detail.append(f'- {r.family}: 直接预测RMSE {r.direct_RMSE_mm:.5f} mm，普通残差 {r.residual_RMSE_mm:.5f} mm，EKI {r.EKI_RMSE_mm:.5f} mm；EKI相对直接预测RMSE变化按“下降”为正计 {r.EKI_vs_direct_RMSE_reduction_percent:.2f}%；技能差 {r.EKI_vs_direct_delta_skill:.4f}，95%区间[{r.delta_skill_CI_low:.4f}, {r.delta_skill_CI_high:.4f}]。')
 text='''# 直接预测—观测残差—完整EKI：补充实验报告

本报告补齐“通用学习器直接预测深度”与完整EKI的同条件对照。A0、A2、A1分别对应直接预测、普通观测残差、完整EKI。只训练100个A0模型，A1/A2及树模型结果经GitHub公开清单哈希核验后复用。

## 主结果（2089目标，15测试结构，结构等权）
'''+mdtable(main)+'\n\n'+'\n'.join(detail)+f'\n\n新增四骨干Holm比较合取：{"通过" if decision["passed_all_four"] else "未通过"}。这项结果仅属于新增探索性比较；原研究P1通过、P2未通过的结论不变。所有差异和不利场景均保留。\n\n'+'''## 科学解释与写作边界

主图与主表可用于检验“模块化工程知识嵌入在通用骨干上的总体作用”。A1对A0包含输出参数化、结构表示及工程响应组织的整体差异，不能解释为每个组件均有效。
A2对A0描述最新观测参考的作用；A1对A2描述在已使用观测参考后工程设计的附加效果。原4.03%–14.36%的数值仍只对应A1对A2，不能改标为A1对A0。
普通直接基线也获得相同历史观测、温度、荷载、结构和间隔时长，故比较不是“不给基线观测”。三者均各自独立训练；骨干类型和训练预算匹配。
这是同一场地、已多次分析的数据上的进一步冻结补充，外层结构留出并不消除研究过程中反复查看数据的影响。没有外部新数据，也没有与人类专家的性能比较。标题中的专家化只指领域知识嵌入。

## 工程场景与区间

scenario_summary.csv包含实际温度、历史同期温度、近期温度递推与一次缺测；late_load_summary.csv包含累计荷载75%以后结果。温度场景未来荷载均相同且给定，严格同期可用集合及回顾性LOYO保持原规则。逐项解释时必须同时查看结构数与目标数。
interval_summary.csv及interval_structures.csv列出平均/最差结构覆盖率、MPIW、MIS95和有效数量。内层校准与选模重复使用同一验证结构，无分布无关覆盖保证。缺测或长递推的精度/区间恶化不能删除。
新增主比较不重新验证EKI形状保证，也不把固定历史的单调性当作真实物理机理识别；已有形状和组件证据按原文保留。

## 建议论文安排

主结果放“三种版本×四骨干”表和图01；图02展示严格历史同期温度、同一起点集合的递推。用A1对A0回答总体增强，用A2桥接解释观测参考贡献；原温度/结构/形状、类别迁移与解释稳定性保留原证据边界。
公式仍区分直接预测、观测加自由增量、观测加上下文变化与温载工程响应。不能将当前数据支持的骨干复用扩大为任意任务或任意模型通用。

## 复现与核验

DESIGN.md、protocol.json、run_matrix.csv、training_freeze.json和analysis_freeze.json固定本轮范围。checks/reuse_audit.json说明前期直接版本不能直接复用的原因（94维与87维输入/头差异）。checks/source_reuse.json绑定GitHub公开数据与既有Linux原件。
checks/independent_supplement_verification.json包含100模型独立重放、主表/结构bootstrap/Holm及内层校准重算。全部种子、学习曲线、失败日志与逐目标场景记录保留。图像仍须人工实际查看后才标记全部交付完成。
'''
 (ROOT/'REPORT_ZH.md').write_text(text,encoding='utf-8');(ROOT/'RESULT_INDEX.md').write_text('# 补充实验结果索引\n\n- DESIGN.md：冻结设计\n- REPORT_ZH.md：科学解释与修稿建议\n- results/tables/three_level_main_table.csv：三层对照主表\n- results/tables/one_step_summary.csv：全部一步指标及2164敏感性\n- results/tables/paired_contrasts.csv：结构配对区间\n- results/tables/supplementary_primary_contrasts.csv：新增四项Holm结果\n- results/tables/scenario_summary.csv：全部递推/缺测指标\n- results/tables/interval_summary.csv及interval_structures.csv：区间汇总与逐结构\n- results/tables/late_load_summary.csv：后段荷载\n- results/tables/new_compute_summary.csv：新增训练资源\n- checks/independent_supplement_verification.json：核验凭据\n- figures/：可导出的PNG/PDF\n',encoding='utf-8')
 write(ROOT/'checks/report_ready.json',{'time':stamp(),'passed':True,'visual_review':'pending','figures':2})
if __name__=='__main__':main()
