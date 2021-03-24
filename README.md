# tbi-pvs-pipe
A pipeline for repeated data preprocessing procedures for different cohorts

## Summary

The contents include how to preprocess data from compressed dicom scans and csv summaries. It is a shared script directory and would **NOT** contain any algorithms or dataset which had not been public released. I would also upload a .pipe file, which is an automated graphical version for LONI Pipeline users, to preprocess their data

## What is New
- Now the pipeline is portable for all directories, and all absolute path requirement has been removed
- QIT and NiftyTorch are introduced for more flexible algorithms

## Installation Requirement
- Python with scientific environment (pandas and pytorch required)
- FreeSurfer v5.3.0 (Please visit http://ftp.nmr.mgh.harvard.edu/pub/dist/freesurfer/5.3.0/)
- FSL v5.0.11 (Please visit https://fsl.fmrib.ox.ac.uk/fsldownloads/patches/eddy-patch-fsl-5.0.11/)
- QIT (Please visit http://cabeen.io/qitwiki/install/)
