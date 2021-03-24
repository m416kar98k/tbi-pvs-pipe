cd ${1}
cd ${2}

qitdiff \
  --motion \
  --subject qitout \
  --dwi ./dmri/dwi.nii.gz \
  --bvecs ./dmri/dwi.bvec \
  --bvals ./dmri/dwi.bval \
  ./diff.regions/jhu.labels.dti.map
