cd ${1}

atlasF="/usr/local/NeuroBattery/data/template/PTBP"

for i in *
do
	mkdir -p ${i}/ants
	qsubcmd sh ${ANTSPATH}/antsCorticalThickness.sh -d 3 \
		-a ${PWD}/${s}/smri/t1w.nii.gz  \
		-e ${atlas_folder}/PTBP_T1_Defaced.nii.gz \
		-m ${atlas_folder}/PTBP_T1_BrainCerebellumProbabilityMask.nii.gz \
		-f ${atlas_folder}/PTBP_T1_ExtractionMask.nii.gz \
		-p ${atlas_folder}/Priors/priors%d.nii.gz \
		-t ${atlas_folder}/PTBP_T1_BrainCerebellum.nii.gz \
		-k 1 -n 3 -w 0.25 \
		-o ${i}/ants/
done
