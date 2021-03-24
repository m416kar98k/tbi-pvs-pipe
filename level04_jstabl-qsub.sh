cd ${1}

for i in *
do
  if [ -d "$i/fsout" ] && [ -f "$i/smri/flair.nii.gz" ]; then
    qsubcmd bash ./jstabl_subject.sh $1 $i
  fi
done
