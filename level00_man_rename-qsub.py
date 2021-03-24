#############################################################
# Level 0 rename
# scripts includes:
# 	- dcm to nifti 

# INI Microstructural imaging Group (IMG)
# Quantitative Imaging Team (QI-Team)
# Steven Neuroimaging and Informatic Institute 
# Keck school of medicine of USC
#############################################################

import os
import pandas as pd

# cohort name
cohort_name = input("Cohort name: ")
while modality not in ("t1w", "t2w", "flair", "dwi"):
	modality = input("Modality name: ")
if modality in ("t1w", "t2w", "flair"):
	modality_type =  "smri"
elif modality in ("dwi"):
	modality_type = "dmri"

# demographic tables
csv_path = input("Demographic table directory: ")
csv_list = os.listdir(csv_path)

# raw compressed dicom files
zip_path = input("Raw scan directory: ")
zip_list = os.listdir(zip_path)

# output
analysisF = input("Analysis directory: ")

# create a dictionary for converting image filename to subjectID and date/visit
subj_list = []
date_list = []
imag_list = []
subj_visit = {}
imag_to_subj = {}

for i in csv_list:
	cur_path = csv_path + "/" + i
	df = pd.read_csv(cur_path, header = 1)
	subj_list += list(df["Main.GUID"])
	date_list += list(df["Main.VisitDate"])
	imag_list += list(df["Image Information.ImgFile"]])

imag_list = list(map(lambda x: x.split("/")[-1], imag_list))
subj_date_list = [cohort_name + "_" + i + "_" + j if isinstance(i, str) and isinstance(j, str) else None for (i,j) in zip(subj_list, date_list)]
for i in range(len(subj_date_list)):
	if isinstance(subj_date_list[i], str):
		if subj_date_list[i] in subj_visit:
			subj_visit[subj_date_list[i]] = 1
		else:
			subj_visit[subj_date_list[i]] += 1
		imag_to_subj[imag_list[i]] = subj_date_list[i] + "_v" + str(subj_visit[subj_date_list[i]])

# submit commands
for i in zip_list:
	if i in imag_to_subj:
		# omit missed data
		var1 = analysisF
		var2 = imag_to_subj[i]
		var3 = zip_path + "/" + i
		var4 = modality
		var5 = modality_type
		cmd = "qsubcmd bash ./level00_man-subject.sh " + var1 + " " + var2 + " " + var3 + " " + var4 + " " + var5
		os.system(cmd)
