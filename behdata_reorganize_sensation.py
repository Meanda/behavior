# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:42:31 2021

@author: zhang
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sns


filepath1 = "E:/tfMRI_GAMBLING/hcpd/tlbx_sensation01.txt"
filepath2 = 'E:/tfMRI_GAMBLING/hcpd/vision_tests01.txt'
parsed1 = pd.read_table(filepath1)
parsed2 = pd.read_table(filepath2)
new_sensation = pd.DataFrame()
vision = pd.DataFrame()
columns1 = parsed1.columns.values.tolist()
columns2 = parsed2.columns.values.tolist()
#for i, column in enumerate(columns2):
#   print(i, column)
column_dic = dict()
num = [4,6,46,55,56,57,58,59]
num_v = [4,12,15]
column_v_dic = dict()
for i, column in enumerate(columns1):
    if i in num:
        column_dic[column] = i
for i, column in enumerate(columns2):
    if i in num_v:
        column_v_dic[column] = i
        
x = 0
for key, value in column_dic.items():
    new_sensation.insert(x, f"{key}", parsed1[columns1[value]])
    x= x+1
    
for key, value in column_v_dic.items():
    vision[key] = parsed2[columns2[value]]

#df.insert(0,'series1',series1)
new_sensation = new_sensation.drop(new_sensation.index[0]).fillna('0')
for key, value in column_dic.items():
    if value != 4:
        new_sensation[columns1[value]] = pd.Series(new_sensation[columns1[value]],dtype=np.float)
aggregation_functions = {'pedsodoridscore': 'sum',
                         'vbdva_stattestscore': 'sum', 
                         'winleft_ncorr':'sum',
                         'winleft_thresholdscore':'sum',
                         'winright_ncorr':'sum',
                         'winright_thresholdscore':'sum',
                         'src_subject_id':'first',
                         'interview_age':'first'}
new_sensation_re = new_sensation.groupby('src_subject_id', as_index=False).aggregate(aggregation_functions).reindex(columns=new_sensation.columns)
#overall = new_sensation_re.describe()
vision = vision.drop(vision.index[0]).fillna('0')
for key, value in column_v_dic.items():
    if value != 4:
        vision[columns2[value]] = pd.Series(vision[columns2[value]],dtype=np.float)
new_sensation_re = pd.merge(new_sensation_re,vision, how='inner',on='src_subject_id')

odor = pd.DataFrame()
visual = pd.DataFrame()
pain = pd.DataFrame()
l_cor_listen = pd.DataFrame()
l_thre_listen = pd.DataFrame()
r_cor_listen = pd.DataFrame()
r_thre_listen = pd.DataFrame()
colorv = pd.DataFrame()
contrastv = pd.DataFrame()
sen_list = [odor, visual, l_cor_listen, l_thre_listen, r_cor_listen, r_thre_listen, colorv, contrastv]
sen_name = ['pedsodoridscore','vbdva_stattestscore', 'winleft_ncorr','winleft_thresholdscore','winright_ncorr','winright_thresholdscore','colorvsn4','contrastvsn3']
odor_od = pd.DataFrame()
vi_od = pd.DataFrame()
l_cor_od = pd.DataFrame()
l_thre_od = pd.DataFrame()
r_cor_od = pd.DataFrame()
r_thre_od = pd.DataFrame()
colorv_od = pd.DataFrame()
contrastv_od = pd.DataFrame()
od_list = [odor_od, vi_od, l_cor_od, l_thre_od, r_cor_od, r_thre_od, colorv_od, contrastv_od]

for i in sen_list:
    i['interview_age'] = new_sensation_re['interview_age']
for x,y  in enumerate(sen_name):
    for m, n in enumerate(sen_list):
        if x == m:
            n[y] = new_sensation_re[y]

#去极端值
for ind, sen in enumerate(sen_list):
    z_score=(sen.iloc[:,1] - sen.iloc[:,1].mean()) / sen.iloc[:,1].std()
    od_list[ind] = sen[z_score.abs()<=3]
sensation_no_od = new_sensation_re.copy()
sensation_od = new_sensation_re.copy()
sen_zscore = new_sensation_re.copy()
cols = sensation_no_od.columns
for col in cols[2:]:
    z_score = (sensation_no_od[col] - sensation_no_od[col].mean()) / sensation_no_od[col].std()
    sen_zscore[col] = z_score.abs()
sensation_od = sensation_no_od[(sen_zscore.iloc[:,2:]>3).any(1)]
sensation_no_od = sensation_no_od[(sen_zscore.iloc[:,2:]<=3).all(1)] #激进型去极端值    
sensation_no_od = sensation_no_od[~sensation_no_od.isin([0]).dropna(axis=0)] #激进型去缺失值

#缺失值分布
sns.distplot(x=sensation_od.iloc[:,1])
mp.show()
mp.close()

#色盲分布
sns.distplot(x=sensation_no_od.iloc[:,8])
mp.xlim(0,4)
mp.show()

sensation_no_od = sensation_no_od.drop('colorvsn4',axis=1)

# 总体pairplot
sns.set(style='darkgrid')
sns.pairplot(sensation_no_od.iloc[:,1:], kind='reg', diag_kind='kde')
mp.savefig('E:/tfMRI_GAMBLING/hcpd/sensation/pairplot/total.jpg')
mp.close()
# 总体相关矩阵
sensation_corr = sensation_no_od.corr()
for i in range(8):
    sensation_corr.iloc[i,i] = np.nan
sns.heatmap(sensation_corr, center=0,annot=True, cmap='YlGnBu')
mp.savefig("E:/tfMRI_GAMBLING/hcpd/sensation/corr_heatmap/total.jpg")

filepath_bp = "E:/tfMRI_GAMBLING/hcpd/sensation/boxplot/{}.jpg"    
filepath_sc = "E:/tfMRI_GAMBLING/hcpd/sensation/scatterplot/{}.jpg"        
for x,y in enumerate(od_list):
    y = y[~y.isin([0])].dropna(axis=0) #去缺失值
    for m, n in enumerate(sen_name):
        if x==m:
            #箱型图
            sns.boxplot(y=y[n])
            mp.savefig(filepath_bp.format(n))
            mp.close()
            #年龄横坐标，行为数据纵坐标，绘制散点图
            sns.set(style='darkgrid') #设置风格
            sns.jointplot(x=y['interview_age'],y=y[n], kind='reg', color='m')
            mp.savefig(filepath_sc.format(n))
            mp.close()

#对听力做单独的平均和偏好分析
listen = pd.DataFrame()
listen['interview_age'] = sensation_no_od.iloc[:,1]
listen['left'] = sensation_no_od.iloc[:,5]
listen['right'] = sensation_no_od.iloc[:,7]
listen['listen_avg'] = listen.iloc[:,1:].mean(axis=1)
listen['listen_bias'] = listen.iloc[:,2] - listen.iloc[:,1]
sns.set(style='darkgrid')
sns.jointplot(x=listen['interview_age'], y=listen.iloc[:,3], kind='reg', color='m')
mp.savefig("E:/tfMRI_GAMBLING/hcpd/sensation/scatterplot/listen_mean.jpg")
mp.close()

sns.set(style='darkgrid')
sns.jointplot(x=listen['interview_age'], y=listen.iloc[:,4]*2/listen.iloc[:,3], kind='reg', color='m')
mp.savefig("E:/tfMRI_GAMBLING/hcpd/sensation/scatterplot/listen_bias.jpg")
mp.close()

sensation_no_od.iloc[:,1] = (sensation_no_od.iloc[:,1]/12).astype('int32')
sen_age_7 = sensation_no_od.loc[sensation_no_od['interview_age']<=7]
age_dict = locals()
age_ddict = {}
corr_dict = {}
for i in range (8,22):
    age_dict[f'sen_age_{str(i)}'] = i
m = 7
for key,value in age_dict.items():
    if 'sen_age' in key:
        age_ddict[key] = sensation_no_od.loc[sensation_no_od['interview_age']==m]
        m += 1
age_ddict['sen_age_7'] = sensation_no_od.loc[sensation_no_od['interview_age']<=7]

for i in range(8,22):
    age_dict[f'corr_age_{str(i)}'] = i

filepath_corr = "E:/tfMRI_GAMBLING/hcpd/sensation/corr_heatmap/{}.jpg"
filepath_pair = "E:/tfMRI_GAMBLING/hcpd/sensation/pairplot/{}.jpg"   
corr_age = pd.DataFrame()
for key, value in age_ddict.items():
    # 相关矩阵 年龄区间内
    non_age = value.iloc[:,2:]
    corr = non_age.corr()
    for i in range(6):
        corr.iloc[i,i] = np.nan
    sns.heatmap(corr, center=0, annot=True, cmap='YlGnBu')
    mp.savefig(filepath_corr.format(key))
    mp.close()
    # pairplot 年龄区间内
    sns.set(style='darkgrid')
    sns.pairplot(non_age, kind='reg', diag_kind='kde')
    mp.savefig(filepath_pair.format(key))
    mp.close()
    

"""    
#画出不同项之间相关值随年龄的发展
full_m = np.empty(shape=(36,0))
for key, value in age_ddict.items():
    non_age = value.iloc[:,2:]
    corr = non_age.corr()
    for i in range(6):
        corr.iloc[i,0:i+1] = np.nan
    sen_matrix = corr.iloc[:,:].values
    sen_matrix = sen_matrix.flatten()
    full_m = np.append(full_m,sen_matrix)
full_m = full_m.reshape(15,36)
full_m = pd.DataFrame(full_m)
full_m = full_m.dropna(axis=1)
full_m.columns = ['odor&vision', 'odor&l_cor','odor&l_thre','odor&r_cor','odor&r_thre',
                  'vision&l_cor','vision&l_thre','vision&r_cor','vision&r_thre',
                  'l_cor&l_thre','l_cor&r_cor','l_cor&r_thre',
                  'l_thre&r_cor','l_thre&r_thre',
                  'r_cor&r_thre']
age_list = list()
for i in range(7,22):
    age_list.append(i)
full_m['age_bin'] = age_list
full_m = full_m.set_index('age_bin')
m_columns = full_m.columns
m_num = [0,2,4,6,8,13]
full_m1 = full_m.iloc[:,m_num]
mp.style.use('ggplot')
sns.lineplot(data=full_m1)
filepath_line = "E:/tfMRI_GAMBLING/hcpd/lineplot/{}.jpg"   
mp.savefig(filepath_line)

"""
  



    


    

    




        
            
            
  



    






    
    
    

    
    









#odfdor_re[f'age_{5}'] = (odor.loc[odor['interview_age']==5]).iloc[:,1]
 
