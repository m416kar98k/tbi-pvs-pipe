

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
