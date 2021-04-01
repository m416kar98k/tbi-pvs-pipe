#############################################################
# Level 2 roi analysis 
# scripts includes:
# 	- run freesurfer on every subject 

# Quantitative Imaging Team (QI-Team)
# INI Microstructural imaging Group (IMG)
# Steven Neuroimaging and Informatic Institute 
# Keck school of medicine of USC
#############################################################

# recall var1 is the analysis absolute path
for i in ${1}/*
do
	if [ ! -d "${i}/fsout" ]
	then
  		qsubcmd bash ./level02_morph_freesurfer-subject.sh ${i}
	fi
done
