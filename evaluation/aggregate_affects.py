import pandas as pd
df = pd.read_csv('affect_results_per_video.csv')
df_other = pd.read_csv('summary_results_generated_final.csv')

#df = df.rename(columns={'valence_diff':'valence_error', 'diff_arousal':'arousal_error'})

df['valence_change'] = df['pred_valence'] - df['src_gt_valence']
df['arousal_change'] = df['pred_arousal'] - df['src_gt_arousal']

df['baseline_valence_error'] = df['src_gt_valence'] - df['target_gt_valence']
df['baseline_arousal_error'] = df['src_gt_arousal'] - df['target_gt_arousal']


df['baseline_valence_change'] = df['target_gt_valence'] - df['src_gt_valence']
df['baseline_arousal_change'] = df['target_gt_arousal'] - df['src_gt_arousal']

df['valence_change_normalized'] = df['valence_change'] / df['baseline_valence_change']
df['arousal_change_normalized'] = df['arousal_change'] / df['baseline_arousal_change']

# penalize *2n modifications that overshoot neutral
towards_neutral = set(('h2n','s2n'))
df['valence_change_normalized'][df['name'].isin(towards_neutral)] = 1 - abs(1 - df['valence_change_normalized'][df['name'].isin(towards_neutral)])
# df['arousal_change_normalized'][df['name'].isin(towards_neutral)] = 1 - abs(1 - df['arousal_change_normalized'][df['name'].isin(towards_neutral)])

df_mean = df.groupby(['name', 'mask','l1_only', 'src_affect_only']).mean()
df_mean[['valence_change_normalized', 'arousal_change_normalized',\
	'valence_change', 'arousal_change', \
	'baseline_valence_change', 'baseline_arousal_change']].to_csv('affect_results.csv')

df_mean_per_affect = df.groupby(['name']).mean()
df_other_per_affect = df_other.groupby(['name']).mean()
df_final_per_affect = pd.concat((df_other_per_affect, df_mean_per_affect), axis=1)
df_final_per_affect = df_final_per_affect.applymap(lambda x: round(x,3))
df_final_per_affect.to_csv('results_final_per_affect.csv')

df_mean_per_affect = df[(df['l1_only']==0)&(df['src_affect_only']==0)&(df['mask']=='half')].groupby(['name']).mean()
df_other_per_affect = df_other[(df_other['l1_only']==0)&(df_other['src_affect_only']==0)&(df_other['mask']=='half')].groupby(['name']).mean()
df_final_per_affect = pd.concat((df_other_per_affect, df_mean_per_affect), axis=1)
df_final_per_affect = df_final_per_affect.applymap(lambda x: round(x,3))
df_final_per_affect.to_csv('results_final_per_affect_best.csv')

df_top_level = df_mean.groupby(['mask','l1_only', 'src_affect_only']).mean()
df_other_top_level = df_other.groupby(['mask','l1_only', 'src_affect_only']).mean()
df_final = pd.concat((df_other_top_level, df_top_level[['valence_change_normalized', 'arousal_change_normalized']]),\
    axis=1)
df_final = df_final.applymap(lambda x: round(x,3))
df_final = df_final.reset_index()[['mask','l1_only', 'src_affect_only', 'LSE-D','LSE-C','FID','valence_change_normalized','arousal_change_normalized']]
df_final[['l1_only', 'src_affect_only']] = df_final[['l1_only', 'src_affect_only']].astype(bool)
df_final.to_csv('results_final.csv',index=False)

