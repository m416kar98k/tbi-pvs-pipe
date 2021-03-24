# recall var1 is the zip absolute path
# recall var2 is the formatted subject name
# recall var3 is the modality
# recall var4 is the modality type
# recall var5 is the analysis absolute path

## enter the analysis
cd ${5}
# make a directory for the subject
mkdir ./${2}
# enter the subject
cd ${2}
# make a raw directory for storing decompressing dicom files
mkdir ./raw
# make an output directory for storing the converted modality
mkdir ./${4}
unzip -q ${1} -d ./raw
dcm2niix -z y -b y -x n -t n -m n -f ${3} -o ./${4} -s n -v n ./raw
## clean the cache
rm -rf ./raw
