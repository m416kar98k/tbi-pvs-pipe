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
cd ${1}
for i in *
do 
	if [ ! -f $i/smri/t1w.orig.nii.gz ];then
    		# backup the scans
		qsubcmd mv ${i}/smri/t1w.nii.gz ${i}/smri/t1w.orig.nii.gz
    		qsubcmd mv ${i}/smri/t2w.nii.gz ${i}/smri/t2w.orig.nii.gz
		qsubcmd mv ${i}/smri/flair.nii.gz ${i}/smri/flair.orig.nii.gz
		qsubcmd mv ${i}/dmri/dwi.bvec ${i}/dmri/dwi.orig.bvec
		qsubcmd mv ${i}/dmri/dwi.nii.gz ${i}/dmri/dwi.orig.nii.gz
		
    		# standardize the scans
		qsubcmd qit VolumeStandardize --input ${i}/smri/t1w.orig.nii.gz --output ${i}/smri/t1w.nii.gz
    		qsubcmd qit VolumeStandardize --input ${i}/smri/t2w.orig.nii.gz --output ${i}/smri/t2w.nii.gz
		qsubcmd qit VolumeStandardize --input ${i}/smri/flair.orig.nii.gz --output ${i}/smri/flair.nii.gz
		qsubcmd qit VolumeStandardize --input ${i}/dmri/dwi.orig.nii.gz --output ${i}/dmri/dwi.nii.gz --xfm ${i}/dmri/nii/dwi.xfm.txt
		qsubcmd qit GradientsTransform --flip x --input ${i}/dmri/nii/dwi.orig.bvec --output ${i}/dmri/input/dwi.bvec
	fi
done
