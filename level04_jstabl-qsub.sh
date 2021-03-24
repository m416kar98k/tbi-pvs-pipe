#############################################################
# Level 4 tumor detection
# scripts includes:
# 	- white matter

# Quantitative Imaging Team (QI-Team)
# INI Microstructural imaging Group (IMG)
# Steven Neuroimaging and Informatic Institute 
# Keck school of medicine of USC
#############################################################

# recall var1 is the  analysis absolute path
cd ${1}

for i in *
do
  if [ -d "$i/fsout" ] && [ -f "$i/smri/flair.nii.gz" ] && [ -f "$i/jstabl/skullstripping/T1_mask.nii.gz" ]; then
    qsubcmd bash ./jstabl_subject.sh $1 $i
  fi
done
