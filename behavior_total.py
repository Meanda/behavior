# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 21:43:56 2021

@author: zhang
"""

import pandas as pd
import numpy as np
#load df
final_path = 'E:/hcpd/hcpd_simplified/final_simplified.csv'
final = pd.read_csv(final_path,header=[0,1])
#name multiindexes and filter different indexes to generate new subtables
final.columns.names = ['categary','index']
raw_cols = final.filter(like='raw')
t_cols = final.filter(like='tscore')
corrected_cols = final.filter(like='corrected')
#to csv
sim_dir = 'E:/hcpd/hcpd_simplified/'
table_dict = {'simplified_raw':raw_cols, 'simplified_t':t_cols, 'simplified_corrected':corrected_cols}
for name,df in table_dict.items():
    df.to_csv(sim_dir+f'{name}.csv',index=False)


