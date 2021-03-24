#############################################################
# Level 2 roi analysis 
# scripts includes:
# 	- run qitdiff on every subject 

# Zhuocheng Li - zhuocheng.li@loni.usc.edu
# INI Microstructural imaging Group (IMG)
# Quantitative Imaging Team (QI-Team)
# Steven Neuroimaging and Informatic Institute 
# Keck school of medicine of USC
#############################################################

# recall var1 is the analysis absolute path
# recall var2 is the formatted subject name

cd ${1}
cd ${2}
qitdiff \
  --motion \
  --subject qitout \
  --dwi ./dmri/dwi.nii.gz \
  --bvecs ./dmri/dwi.bvec \
  --bvals ./dmri/dwi.bval \
  ./diff.regions/jhu.labels.dti.map
