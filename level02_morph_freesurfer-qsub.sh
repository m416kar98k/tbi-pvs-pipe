#############################################################
# Level 2 roi analysis 
# scripts includes:
# 	- run freesurfer on every subject 

# Zhuocheng Li - zhuocheng.li@loni.usc.edu
# INI Microstructural imaging Group (IMG)
# Quantitative Imaging Team (QI-Team)
# Steven Neuroimaging and Informatic Institute 
# Keck school of medicine of USC
#############################################################

# recall var1 is the analysis absolute path
cd ${1}
for i in *
do
	if [ ! -f "${data_folder}/${subName}/fsout/stats/aseg.stats" ]; then
  		qsubcmd bash ./level02_morph_freesurfer-subject.sh ${1} ${i}
	fi
done
