#############################################################
# Level 2 segmentation thickness
# scripts includes:
# 	- gray matter 

# Quantitative Imaging Team (QI-Team)
# INI Microstructural imaging Group (IMG)
# Steven Neuroimaging and Informatic Institute 
# Keck school of medicine of USC
#############################################################

# recall var1 is the analysis absolute path

atlasF="/usr/local/NeuroBattery/data/template/PTBP"

for i in ${1}/*
do
	mkdir -p ${i}/ants
	qsubcmd sh antsCorticalThickness.sh -d 3 \
		-a ${PWD}/${s}/smri/t1w.nii.gz  \
		-e ${atlas_folder}/PTBP_T1_Defaced.nii.gz \
		-m ${atlas_folder}/PTBP_T1_BrainCerebellumProbabilityMask.nii.gz \
		-f ${atlas_folder}/PTBP_T1_ExtractionMask.nii.gz \
		-p ${atlas_folder}/Priors/priors%d.nii.gz \
		-t ${atlas_folder}/PTBP_T1_BrainCerebellum.nii.gz \
		-k 1 -n 3 -w 0.25 \
		-o ${i}/ants/
done
