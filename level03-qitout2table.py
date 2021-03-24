#############################################################
# Level 3 result to table
# scripts includes:
# 	- qitout to csv

# Zhuocheng Li - zhuocheng.li@loni.usc.edu
# INI Microstructural imaging Group (IMG)
# Quantitative Imaging Team (QI-Team)
# Steven Neuroimaging and Informatic Institute 
# Keck school of medicine of USC
#############################################################

import os
import pandas as pd

# output
analysisF = input("Analysis directory: ")
statF = input("Statistics directory: ")

df2 = []

subj_list = os.listdir(analysisF)
for i in subj_list:
	column_names = []
	csv_path = i + "/" + "qitout" + "/" + "diff.regions" + "/" + "jhu.labels.dti.map"
	if os.path.isdir(csv_path):
		column_names += [i.replace(".csv", "") + "_" + k for k in temp["name"]]
		for j in os.listdir(csv_path):
			temp = pd.read_csv(csv_path + "/" + j)
			df2.append([i] + list(temp["value"]))

pd.DataFrame(df2, columns = ["Subject"] + column_names).to_csv(statF + "/" + "qitout.csv")
