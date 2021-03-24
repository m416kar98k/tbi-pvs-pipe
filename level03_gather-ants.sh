#############################################################
# Level 3 segmentation analysis
# scripts includes:
# 	- gather global measures from ANTs 

# Quantitative Imaging Team (QI-Team)
# INI Microstructural imaging Group (IMG)
# Steven Neuroimaging and Informatic Institute 
# Keck school of medicine of USC
#############################################################

# recall var1 is the analysis absolute path
# recall var2 is the statistics absolute path
cd ${1}

echo "Subject,BVol,GVol,WVol,ThicknessSum" > ${2}/ants.all.csv

for i in *
do
	BVol=`cat ${i}/ants/brainvols.csv | tail -1 | cut -d ',' -f2`
	GVol=`cat ${i}/ants/brainvols.csv | tail -1 | cut -d ',' -f3`
	WVol=`cat ${i}/ants/brainvols.csv | tail -1 | cut -d ',' -f4`
	ThicknessSum=`cat ${i}/ants/brainvols.csv | tail -1 | cut -d ',' -f5`

	echo "${i},${BVol},${GVol},${WVol},${ThicknessSum}" >> ${2}/ants.all.csv
done
