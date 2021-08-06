#!/usr/bin/env python
import sys
import os
df2 = []
for subject in sorted(os.listdir(sys.argv[1])):
 for visit in sorted(os.listdir(sys.argv[1]+"/"+subject+"/raw")):
  os.system("qsubcmd qitdiff --motion --subject "+sys.argv[1]+"/"+subject+"/qitout  --dwi "+sys.argv[1]+"/"+subject+"/raw/"+visit+"/dmri/dwi.nii.gz --bvecs "+sys.argv[1]+"/"+subject+"/raw/"+visit+"/dmri/dwi.bvec --bvals "+sys.argv[1]+"/"+subject+"/raw/"+visit+"/dmri/dwi.bval tone.region/fs.ccwm.dti.map tone.region/fs.scgm.dti.map tone.region/fs.dkwm.dti.map tone.region/fs.dkgm.dti.map")
