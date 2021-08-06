# preppipe
那里讨烟蓑雨笠卷单行 一任俺芒鞋破钵随缘化
![mmexport1627849541133](https://user-images.githubusercontent.com/70918897/128453629-2f1fc00d-bef2-4b22-9bff-111856d15457.jpg)

## Recent Update
- remove LONI Pipeline

## Next Update
- Add Longitudinal Freesurfer
- Change QIT from JHU to Freesurfer
- Add DeepGBM Pipeline

## Installation Requirement
- Python with scientific environment (pandas and pytorch required)
- ANTs (Please visit https://github.com/ANTsX/ANTs/)
- FreeSurfer v7.2.0 (Please visit http://ftp.nmr.mgh.harvard.edu/pub/dist/freesurfer/dev/)
- FSL v6.0.4 (Please visit https://fsl.fmrib.ox.ac.uk/fsldownloads/)
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
