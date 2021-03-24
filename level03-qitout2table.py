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
	csv_path = analysisF + "/" + i + "/" + "qitout" + "/" + "diff.regions" + "/" + "jhu.labels.dti.map"
	if os.path.isdir(csv_path):
		column_names = []
		column_values = [i]
		for j in os.listdir(csv_path):
			temp = pd.read_csv(csv_path + "/" + j)
			column_names += [j.replace(".csv", "") + "_" + k for k in temp["name"]]
			column_values += list(temp)
		df2.append(column_values)

pd.DataFrame(df2, columns = ["Subject"] + column_names).to_csv(statF + "/" + "qitout.csv")
