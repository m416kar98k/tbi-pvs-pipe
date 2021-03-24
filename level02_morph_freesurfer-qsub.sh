# recall var1 is the analysis absolute path
for i in ${var1}/*
do
	subName=`echo $i | cut -d '/' -f10`
	if [ ! -f "${data_folder}/${subName}/fsout/stats/aseg.stats" ]; then
  		qsubcmd bash ./level02_morph_freesurfer-subject.sh $subName
	fi
done
