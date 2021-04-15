# tbi-pvs-pipe
A pipeline for repeated data preprocessing procedures for different cohorts

## Summary

The contents include how to preprocess data from compressed dicom scans and csv summaries. It is a shared script directory and would **NOT** contain any PVS algorithms or original dataset which had not been public released. I would also upload a .pipe file, which is an automated graphical version for LONI Pipeline users, to preprocess their data

## Installation Requirement
- Python with scientific environment (pandas and pytorch required)
- ANTs (Please visit https://github.com/ANTsX/ANTs/)
- FreeSurfer v5.3.0 (Please visit http://ftp.nmr.mgh.harvard.edu/pub/dist/freesurfer/5.3.0/)
- FSL v5.0.11 (Please visit https://fsl.fmrib.ox.ac.uk/fsldownloads/patches/eddy-patch-fsl-5.0.11/)
- QIT (Please visit http://cabeen.io/qitwiki/install/)
- jSTABL (Please visit https://github.com/ReubenDo/jSTABL/)
- NeuroBattery (Please visit https://github.com/jeffduda/NeuroBattery/)

## How to Run
- Select ROOT_FOLDER your for demographic summary and compressed scanning
- Put your demographic summary in ROOT_FOLDER/csv
- Put your compressed scanning in ROOT_FOLDER/zip

If you are preprocessing structural data:
- run ```python level00_man_rename-qsub.py ${ROOT_FOLDER} TRACKTBI t1w```
- run ```bash level01_preproc-standardize.sh ${ROOT_FOLDER}/analysis```
- run ```bash level02_morph_freesurfer-qsub.sh ${ROOT_FOLDER}/analysis```
- run ```bash level03_fsout2table.sh ${ROOT_FOLDER}/analysis```

If you are preprocessing diffusion data:
1. run ```python level00_man_rename-qsub.py ${ROOT_FOLDER} TRACKTBI dwi```
2. run ```bash level01_preproc-standardize.sh ${ROOT_FOLDER}```
3. run ```bash level02_native_qitdiff-qsub.sh ${ROOT_FOLDER}```
4. run ```bash level03_qitout2table.py ${ROOT_FOLDER}```
