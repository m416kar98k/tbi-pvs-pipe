# recall var1 is the analysis absolute path
cd ${var1}
for i in *
do
	if [ ! -f "${data_folder}/${subName}/fsout/stats/aseg.stats" ]; then
  		qsubcmd bash ./level02_morph_freesurfer-subject.sh ${i}
	fi
done
