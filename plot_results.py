"""Presentation only: every plotted value is read from frozen-analysis tables."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
F=['mlp','gru','lstm','transformer'];LABEL={'mlp':'MLP','gru':'GRU','lstm':'LSTM','transformer':'Transformer'}
C={'A1':'#0072B2','A2':'#E69F00','A3':'#CC79A7','A4':'#009E73','A6':'#D55E00','A7':'#56B4E9'}
NAME={'A1':'EKI','A2':'Residual NN','A3':'Unconstrained','A4':'Residual XGB','A6':'No current T','A7':'Flat structure'}
TABLE=ROOT/'results/tables';FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':7,'legend.frameon':False,'figure.dpi':120,'savefig.dpi':300,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'lines.linewidth':1.5,'lines.markersize':4,'pdf.fonttype':42,'ps.fonttype':42})
CATALOG=[]
def read(n):return pd.read_csv(TABLE/n)
def save(fig,name,caption,sources):
 fig.savefig(FIG/(name+'.pdf'));fig.savefig(FIG/(name+'.png'));plt.close(fig);CATALOG.append(dict(name=name,caption_zh=caption,sources=sources,PNG_SHA256=digest(FIG/(name+'.png')),PDF_SHA256=digest(FIG/(name+'.pdf'))))
def forest(ax,df,labels,boundary=0):
 y=np.arange(len(df));ax.hlines(y,df.skill_CI_low,df.skill_CI_high,color='#0072B2',lw=1.5);ax.plot(df.delta_skill,y,'o',color='#0072B2',ms=4);ax.axvline(boundary,color='#666666',ls='--',lw=1);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('Paired skill difference (descriptive 95% interval)')
def main():
 assert json.loads((ROOT/'checks/independent_verification.json').read_text())['passed']
 one=read('one_step_summary.csv');one=one[(one.scope=='primary2089')&(one.block=='main')];x=np.arange(4);fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
 for ax,metric,title in zip(axes,['skill','macro_RMSE_mm'],['Structure-balanced skill','Structure-balanced RMSE (mm)']):
  for j,arm in enumerate(['A1','A2','A3']):
   a=one[one.arm==arm].set_index('family').reindex(F);ax.bar(x+(j-1)*.22,a[metric],.2,label=NAME[arm],color=C[arm])
  for arm,ls in [('A4','--'),('A5',':')]:
   value=one[one.arm==arm][metric].iloc[0];ax.axhline(value,color=C.get(arm,'#666666'),ls=ls,label=NAME.get(arm,'Persistence'))
  ax.set_xticks(x,[LABEL[f] for f in F]);ax.set_ylabel(title)
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.09),ncol=5);save(fig,'fig01_main','2089目标，五种子先等权集成、结构均衡汇总。柱高是点估计；树与persistence为同目标参照。统计证据见图2及P1/P2表。',['one_step_summary.csv'])
 con=read('primary_backbone_contrasts.csv');fig,axes=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
 for ax,test,boundary in zip(axes,['P1','P2'],[0,-.03]):
  a=con[con.comparison==test].set_index('family').reindex(F).reset_index();forest(ax,a,[LABEL[f] for f in F],boundary);ax.set_title(test+(': EKI - residual NN' if test=='P1' else ': EKI - unconstrained'))
 save(fig,'fig02_primary_contrasts','各骨干的配对技能差与未校正95%结构bootstrap区间。正式决策为四骨干合取后对P1/P2做Holm，不由图中区间单独判断。',['primary_backbone_contrasts.csv','primary_conjunctions.csv'])
 sm=read('scenario_summary.csv');common=sm[sm.scope=='past_common'];fig,axes=plt.subplots(4,3,figsize=(10,9),layout='constrained');scenarios=['realized','calendar_past','recent_temperature'];titles=['Realized temperature','Strict-past calendar','Recent temperature']
 for i,f in enumerate(F):
  for j,scenario in enumerate(scenarios):
   ax=axes[i,j]
   for arm,fam,mark in [('A1',f,'o'),('A2',f,'s'),('A4','xgb','^')]:
    z=common[(common.family==fam)&(common.arm==arm)&(common.scenario==scenario)].sort_values('horizon');ax.plot(z.horizon,z.skill,marker=mark,color=C[arm],label=NAME[arm])
   ax.axhline(0,color='#888888',lw=.6);ax.set_xticks([1,2,3,4]);ax.set_xlabel('Forecast steps');ax.set_ylabel(LABEL[f]+' | skill')
   if i==0:ax.set_title(titles[j])
 axes[0,0].legend();save(fig,'fig03_rollout','相同严格历史温度可用起点、相同给定荷载的1–4步递推。只替换温度，未来车辙均由集成预测更新。所有线均为描述性点估计，实际温度不是必然最好。',['scenario_summary.csv'])
 fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
 for ax,f in zip(axes.flat,F):
  for scenario,label,color,marker in [('realized','Realized','#009E73','o'),('calendar_past','Strict past','#0072B2','s'),('recent_temperature','Recent','#E69F00','^'),('calendar_loyo_retrospective','LOYO retrospective','#CC79A7','D')]:
   z=common[(common.arm=='A1')&(common.family==f)&(common.scenario==scenario)].sort_values('horizon');ax.plot(z.horizon,z.macro_RMSE_mm,label=label,color=color,marker=marker)
  ax.set_title(LABEL[f]);ax.set_xlabel('Forecast steps');ax.set_ylabel('Balanced RMSE (mm)');ax.set_xticks([1,2,3,4])
 axes[0,0].legend();save(fig,'fig04_temperature_sensitivity','EKI在共同起点的温度方案敏感性。LOYO可能使用起点后资料，明确为回顾性，不作为部署证据。',['scenario_summary.csv'])
 iv=read('interval_summary.csv');iv=iv[(iv.scope=='past_common')&(iv.scenario=='calendar_past')];fig,axes=plt.subplots(4,3,figsize=(10,9),layout='constrained')
 for i,f in enumerate(F):
  for j,(metric,label) in enumerate([('mean_structure_coverage','Mean structure coverage'),('MPIW_mm','Mean interval width (mm)'),('MIS95_mm','MIS95 (mm)')]):
   ax=axes[i,j]
   for arm,fam,mark in [('A1',f,'o'),('A2',f,'s'),('A4','xgb','^')]:
    z=iv[(iv.arm==arm)&(iv.family==fam)].sort_values('horizon');ax.plot(z.horizon,z[metric],color=C[arm],marker=mark,label=NAME[arm])
   if j==0:ax.axhline(.95,color='#888888',ls='--');ax.set_ylim(0,1.02)
   ax.set_xticks([1,2,3,4]);ax.set_xlabel('Forecast steps');ax.set_ylabel(label)
   if j==0:ax.set_title(LABEL[f])
 axes[0,0].legend();save(fig,'fig05_interval_tradeoff','严格历史温度递推的经验区间覆盖、宽度与MIS95联合报告。半径只由选中内层checkpoint的结构均衡残差取得；复用选模集且最终refit不同，不保证95%覆盖。',['interval_summary.csv'])
 au=read('response_audit.csv');fig,axes=plt.subplots(2,3,figsize=(10,5.7),layout='constrained')
 for j,p in enumerate(['load10','load50','temp5']):
  for ai,arm in enumerate(['A1','A3']):
   z=au[(au.arm==arm)&(au.perturbation==p)].set_index('family').reindex(F);axes[0,j].bar(x+(ai-.5)*.3,z.violation_fraction*100,.27,color=C[arm],label=NAME[arm]);axes[1,j].plot(x,z.mean_change_mm,'o-' if arm=='A1' else 's--',color=C[arm],label=NAME[arm])
  axes[0,j].set_title({'load10':'Load +10%','load50':'Load +50%','temp5':'Daily temperature +5 °C'}[p]);axes[0,j].set_ylabel('Violations (%)');axes[1,j].set_ylabel('Mean output change (mm)')
  for ax in axes[:,j]:ax.set_xticks(x,[LABEL[f] for f in F],rotation=20);ax.axhline(0,color='#888888',lw=.6)
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.06),ncol=2);save(fig,'fig06_shape_audit','固定历史、结构和时长，对每目标每种子扰动。上排违规比例，下排响应幅度；0违规只证明实现了条件形状约束。完整原始负差及训练范围标记保留在response_rows。',['response_audit.csv'])
 desc=read('descriptive_components_tree.csv');z=desc[desc.comparison.isin(['A1_minus_A6','A1_minus_A7'])];fig,ax=plt.subplots(figsize=(8,3),layout='constrained');forest(ax,z,[LABEL[r.family]+' | '+('current T' if r.comparison.endswith('A6') else 'structure') for r in z.itertuples()]);save(fig,'fig07_components','固定MLP/GRU的温度调制与结构语义消融。区间为描述性，所有方向都保留。A6仍含历史温度，不能称去掉全部温度。',['descriptive_components_tree.csv'])
 cl=read('class_contrasts.csv');fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
 for ax,sp in zip(axes.flat,sorted(cl.split.unique())):
  z=cl[(cl.split==sp)&(cl.comparison=='A1_minus_A7')].set_index('family').reindex(F).reset_index();forest(ax,z,[LABEL[f] for f in F]);ax.set_title(sp.replace('class_','Class '))
 save(fig,'fig08_class_transfer','整类留出中的EKI语义结构减flat结构技能差；只覆盖材料可由训练集支持的I/IV/VI/VII类，每类结构数很少，区间仅作描述。树对照完整结果见表。',['class_contrasts.csv'])
 d1=read('D1_load_synchrony.csv');d2=read('D2_load_correlations.csv');fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
 for definition,color in [('end_day','#0072B2'),('start_day+end_day','#D55E00')]:
  z=d1[(d1.definition==definition)&(d1.structures>=2)];axes[0].scatter(pd.to_datetime(z.end_day,unit='D',origin='unix'),z.span_10k,s=9,alpha=.55,color=color,label=definition)
 axes[0].set_ylabel('Across-structure load range (10k)');axes[0].tick_params(axis='x',rotation=30);axes[0].legend()
 z=d2[(d2.scope=='all19')&(d2.STR_name!='pooled')];piv=z.pivot(index='STR_name',columns='variable',values='Pearson').reindex(sorted(z.STR_name.unique(),key=lambda x:int(x[3:])));im=axes[1].imshow(piv.to_numpy(),aspect='auto',vmin=-1,vmax=1,cmap='coolwarm');axes[1].set_xticks(range(len(piv.columns)),['Mean-T heat' if x=='daily_mean_heat25_proxy' else '10-min heat' if x=='degree_hours25' else 'Days' for x in piv.columns],rotation=20);axes[1].set_yticks(range(len(piv)),piv.index,fontsize=6);fig.colorbar(im,ax=axes[1],label='Within-structure Pearson r')
 save(fig,'fig09_load_data','D1同时给出仅终点日期匹配与完整区间匹配；D2逐结构相关。共同日历、区间长短和加载安排均可能产生相关，不能用这些表证明荷载没有物理作用。',['D1_load_synchrony.csv','D2_load_correlations.csv'])
 at=read('attribution_global.csv');fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained');groups=['memory','load','temperature','structure']
 for ax,f in zip(axes.flat,F):
  for j,arm in enumerate(['A1','A2']):
   z=at[(at.arm_id=='main_'+arm+'_'+f)&(at.reference=='main')].set_index('group').reindex(groups);ax.bar(np.arange(4)+(j-.5)*.3,z.abs_phi_mm,.27,color=C[arm],label=NAME[arm])
  ax.set_xticks(range(4),groups,rotation=20);ax.set_title(LABEL[f]);ax.set_ylabel('Mean absolute group attribution (mm)')
 axes[0,0].legend();save(fig,'fig10_attribution','对增量输出的四组精确Shapley归因，训练侧固定背景，每结构10个无标签预选查询，五种子先平均后取绝对值。归因大小不是消融收益或因果贡献。',['attribution_global.csv'])
 st=read('attribution_seed_stability.csv');bg=read('attribution_background_stability.csv');fig,axes=plt.subplots(1,3,figsize=(10,3.6),layout='constrained')
 for ax,source,col,title in [(axes[0],st[st.reference=='main'],'seed_L1_mm','Across seeds (mm)'),(axes[1],st[st.reference=='main'],'normalized_seed_L1','Across seeds (normalized)'),(axes[2],bg,'normalized_L1','Across backgrounds (normalized)')]:
  for j,arm in enumerate(['A1','A2']):
   z=source.set_index('arm_id').reindex(['main_'+arm+'_'+f for f in F]);ax.bar(x+(j-.5)*.3,z[col],.27,color=C[arm],label=NAME[arm])
  ax.set_xticks(x,[LABEL[f] for f in F],rotation=20);ax.set_ylabel('Attribution variation');ax.set_title(title)
 axes[0].legend();save(fig,'fig11_attribution_stability','同时报告归因原始与归一化跨种子离散、背景变化；稳定不等于正确，幅度小可能导致原始离散较小。',['attribution_seed_stability.csv','attribution_background_stability.csv'])
 write(ROOT/'results/figure_catalog.json',CATALOG);write(ROOT/'checks/plots_generated.json',{'time':stamp(),'figures':len(CATALOG),'source':'frozen numerical tables only','visual_review_pending':True});print('PLOTS_GENERATED',len(CATALOG))
if __name__=='__main__':main()
