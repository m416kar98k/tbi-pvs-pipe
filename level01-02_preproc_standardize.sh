#############################################################
# Level 1 standardize
# scripts includes:
# 	- volume standardize 

# Quantitative Imaging Team (QI-Team)
# INI Microstructural imaging Group (IMG)
# Steven Neuroimaging and Informatic Institute 
# Keck school of medicine of USC
#############################################################

# recall var1 is the analysis absolute path
for i in ${1}/*
do	
  # standardize the scans
	qsubcmd qit VolumeStandardize --input ${i}/smri/t1w.orig.nii.gz --output ${i}/smri/t1w.nii.gz
	qsubcmd qit VolumeStandardize --input ${i}/smri/t2w.orig.nii.gz --output ${i}/smri/t2w.nii.gz
	qsubcmd qit VolumeStandardize --input ${i}/smri/flair.orig.nii.gz --output ${i}/smri/flair.nii.gz
	qsubcmd qit VolumeStandardize --input ${i}/dmri/dwi.orig.nii.gz --output ${i}/dmri/dwi.nii.gz --xfm ${i}/dmri/dwi.xfm.txt
	qsubcmd qit GradientsTransform --flip x --input ${i}/dmri/dwi.orig.bvec --output ${i}/dmri/dwi.bvec
done
