# recall var1 is the analysis absolute path
cd ${1}
for i in *
do
  qsubcmd --qbigmem bash ./level02_native_qitdiff-subject.sh ${1} ${i}
done
