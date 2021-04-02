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
	# backup the scans
	qsubcmd mv ${i}/smri/t1w.nii.gz ${i}/smri/t1w.orig.nii.gz
    	qsubcmd mv ${i}/smri/t2w.nii.gz ${i}/smri/t2w.orig.nii.gz
	qsubcmd mv ${i}/smri/flair.nii.gz ${i}/smri/flair.orig.nii.gz
	qsubcmd mv ${i}/dmri/dwi.nii.gz ${i}/dmri/dwi.orig.nii.gz
	qsubcmd mv ${i}/dmri/dwi.bvec ${i}/dmri/dwi.orig.bvec
done
