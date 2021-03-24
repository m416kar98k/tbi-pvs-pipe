# recall var1 is the analysis absolute path
cd ${1}
for i in *
do 
	if [ ! -f $i/smri/t1w.orig.nii.gz ];then
    		# backup the scans
		qsubcmd cp ${i}/smri/t1w.nii.gz ${i}/smri/t1w.orig.nii.gz
    		qsubcmd cp ${i}/smri/t2w.nii.gz ${i}/smri/t2w.orig.nii.gz
		qsubcmd cp ${i}/smri/flair.nii.gz ${i}/smri/flair.orig.nii.gz
		qsubcmd cp ${i}/dmri/dwi.nii.gz ${i}/dmri/dwi.orig.nii.gz
    		# standardize the scans
		qsubcmd qit VolumeStandardize --input ${i}/smri/t1w.orig.nii.gz --output ${i}/smri/t1w.nii.gz
    		qsubcmd qit VolumeStandardize --input ${i}/smri/t2w.orig.nii.gz --output ${i}/smri/t2w.nii.gz
		qsubcmd qit VolumeStandardize --input ${i}/smri/flair.orig.nii.gz --output ${i}/smri/flair.nii.gz
		qsubcmd qit VolumeStandardize --input ${i}/dmri/dwi.orig.nii.gz --output ${i}/dmri/dwi.nii.gz --xfm ${i}/dmri/nii/dwi.xfm.txt
	fi
done
