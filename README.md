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
- For every level, must run the previous levels to continue on the next level.
- Select CSV_FOLDER and ZIP_FOLDER for demographic summary and compressed scanning
- Select ANALYSIS_FOLDER and STATISTICS_FOLDER

If you are preprocessing structural data:
1. run level00_man_rename-qsub.py
2. run level01_preproc-standardize.sh
3. run level02_morph_freesurfer-qsub.sh
4. run level03_fsout2table.sh

If you are preprocessing diffusion data:
1. run level00_man_rename-qsub.py
2. run level01_preproc-standardize.sh
3. run level02_native_qitdiff-qsub.sh
4. run level03_qitout2table.py
