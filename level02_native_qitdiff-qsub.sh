#############################################################
# Level 2 roi analysis 
# scripts includes:
# 	- run qitdiff on every subject 

# Quantitative Imaging Team (QI-Team)
# INI Microstructural imaging Group (IMG)
# Steven Neuroimaging and Informatic Institute 
# Keck school of medicine of USC
#############################################################

# recall var1 is the analysis absolute path
cd ${1}
for i in *
do
  qsubcmd --qbigmem bash ./level02_native_qitdiff-subject.sh ${1} ${i}
done
