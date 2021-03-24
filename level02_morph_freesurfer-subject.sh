# recall var1 is the analysis absolute path
# recall var2 is the formatted subject name
cd ${1}
unset SUBJECTS_DIR
SUBJECTS_DIR="${2}/smri"
export SUBJECTS_DIR
# define t1 image
t1="${data_folder}/smri/t1w.nii.gz"
## freesurfer bit + modifying the output
recon-all -s ${2} -i ${t1} -all
mkdir -p ${2}/fsout/
mv ${2}/smri/${2}/* ${2}/fsout/
# convert to nifti
mri_convert ${2}/fsout/mri/brainmask.mgz ${2}/smri/t1w.brain.nii.gz
