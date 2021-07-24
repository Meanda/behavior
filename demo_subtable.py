# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 08:18:31 2021

@author: 亚亚
"""
import pandas as pd
import numpy as np
#load the behaviorial directory
sim_path = "D:/hcpd/hcpd_simplified/behavior_simplified_dictionary.xlsx"

df = pd.read_excel(sim_path)
for ind, row in df.iterrows():
    element = row[2].split(',')
    for i in ['version_form', 'comqother']:
        if i in element:
            df.iloc[ind,1] = row[1]+i
            element.remove(i)
        df.iloc[ind,2] = ','.join(element)
df.to_excel("D:/hcpd/hcpd_simplified/behavior_simplified_dictionary.xlsx",index=False)
        
        

        