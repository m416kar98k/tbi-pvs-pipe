#!/usr/bin/env python
import sys
import os
for subject in sorted(os.listdir(sys.argv[1])):
 for visit in sorted(os.listdir(sys.argv[1]+"/"+subject+"/raw")):
  os.system("qsubcmd qitdiff --motion --subject "+sys.argv[1]+"/"+subject+"/qitout --freesurfer "+sys.argv[1]+"/"+subject+"/fsout/"+visit+".long."+subject+" --dwi "+sys.argv[1]+"/"+subject+"/raw/"+visit+"/dmri/dwi.nii.gz --bvecs "+sys.argv[1]+"/"+subject+"/raw/"+visit+"/dmri/dwi.bvec --bvals "+sys.argv[1]+"/"+subject+"/raw/"+visit+"/dmri/dwi.bval tone.fs.map tone.region/fs.ccwm.dti.map tone.region/fs.dkbm.dti.map tone.region/fs.dkgm.dti.map tone.region/fs.dkwm.dti.map tone.region/fs.lbbm.dti.map tone.region/fs.lbgm.dti.map tone.region/fs.lbwm.dti.map tone.region/fs.scgm.dti.map tone.region/fs.wbbm.dti.map")
