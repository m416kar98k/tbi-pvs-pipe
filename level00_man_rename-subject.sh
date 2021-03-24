# recall var1 is the analysis absolute path
# recall var2 is the formatted subject name
# recall var3 is the zip absolute path
# recall var4 is the modality
# recall var5 is the modality type

## enter the analysis
cd ${1}
# create a directory for the subject
mkdir ./${2}
# enter the subject
cd ${2}
# make a raw directory for storing decompressing dicom files
mkdir ./raw
# make an output directory for storing the converted modality
mkdir ./${5}
unzip -q ${3} -d ./raw
dcm2niix -z y -b y -x n -t n -m n -f ${4} -o ./${5} -s n -v n ./raw
## cache
rm -rf ./raw
