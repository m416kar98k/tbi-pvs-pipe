#############################################################
# Level 4 tumor detection
# scripts includes:
# 	- white matter

# Quantitative Imaging Team (QI-Team)
# INI Microstructural imaging Group (IMG)
# Steven Neuroimaging and Informatic Institute 
# Keck school of medicine of USC
#############################################################

# recall var1 is the  analysis absolute path
# recall var2 is the subject
cd ${1}

mkdir -p ${2}/jstabl
mkdir -p ${2}/jstabl/skullstripping
mri_convert -i ${2}/fsout/mri/T1.mgz -o ${2}/jstabl/t1w.nii.gz
bet ${2}/jstabl/t1w.nii.gz ${2}/jstabl/skullstripping/T1 -o -R -m
jstabl_wmh -t1 ${2}/jstabl/t1w.nii.gz -fl ${2}/smri/flair.nii.gz -res ${2}/jstabl/segmentation.nii.gz --preprocess
