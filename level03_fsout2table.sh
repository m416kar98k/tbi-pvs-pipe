#############################################################
# Level 3 result to table
# scripts includes:
# 	- fsout to csv

# Zhuocheng Li - zhuocheng.li@loni.usc.edu
# INI Microstructural imaging Group (IMG)
# Quantitative Imaging Team (QI-Team)
# Steven Neuroimaging and Informatic Institute 
# Keck school of medicine of USC
#############################################################

# recall var1 is the  analysis absolute path
# recall var2 is the stat absolute path
cd ${1}
mkdir ./temp

for i in *
do
  mkdir -p ./temp/${i}
  cp -r ${i}/fsout/stats ./temp/${i}/
done

cd ./temp
SUBJECTS_DIR=${1}/temp
export SUBJECTS_DIR

list=`ls -d *`
python2 $FREESURFER_HOME/bin/asegstats2table --subjects $list --meas volume --skip --statsfile wmparc.stats --all-segs --tablefile wmparc_stats.txt
python2 $FREESURFER_HOME/bin/asegstats2table --subjects $list --meas volume --skip --tablefile aseg_stats.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --subjects $list --hemi lh --meas volume --skip --tablefile aparc_volume_lh.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --subjects $list --hemi lh --meas thickness --skip --tablefile aparc_thickness_lh.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --subjects $list --hemi lh --meas area --skip --tablefile aparc_area_lh.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --subjects $list --hemi lh --meas meancurv --skip --tablefile aparc_meancurv_lh.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --subjects $list --hemi rh --meas volume --skip --tablefile aparc_volume_rh.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --subjects $list --hemi rh --meas thickness --skip --tablefile aparc_thickness_rh.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --subjects $list --hemi rh --meas area --skip --tablefile aparc_area_rh.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --subjects $list --hemi rh --meas meancurv --skip --tablefile aparc_meancurv_rh.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --hemi lh --subjects $list --parc aparc.a2009s --meas volume --skip -t lh.a2009s.volume.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --hemi lh --subjects $list --parc aparc.a2009s --meas thickness --skip -t lh.a2009s.thickness.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --hemi lh --subjects $list --parc aparc.a2009s --meas area --skip -t lh.a2009s.area.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --hemi lh --subjects $list --parc aparc.a2009s --meas meancurv --skip -t lh.a2009s.meancurv.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --hemi rh --subjects $list --parc aparc.a2009s --meas volume --skip -t rh.a2009s.volume.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --hemi rh --subjects $list --parc aparc.a2009s --meas thickness --skip -t rh.a2009s.thickness.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --hemi rh --subjects $list --parc aparc.a2009s --meas area --skip -t rh.a2009s.area.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --hemi rh --subjects $list --parc aparc.a2009s --meas meancurv --skip -t rh.a2009s.meancurv.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --hemi lh --subjects $list --parc BA --meas volume --skip -t lh.BA.volume.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --hemi lh --subjects $list --parc BA --meas thickness --skip -t lh.BA.thickness.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --hemi lh --subjects $list --parc BA --meas area --skip -t lh.BA.area.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --hemi lh --subjects $list --parc BA --meas meancurv --skip -t lh.BA.meancurv.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --hemi rh --subjects $list --parc BA --meas volume --skip -t rh.BA.volume.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --hemi rh --subjects $list --parc BA --meas thickness --skip -t rh.BA.thickness.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --hemi rh --subjects $list --parc BA --meas area --skip -t rh.BA.area.txt
python2 $FREESURFER_HOME/bin/aparcstats2table --hemi rh --subjects $list --parc BA --meas meancurv --skip -t rh.BA.meancurv.txt

mkdir -p ${2}/temp
mkdir -p ${2}/fsout

mv *.txt ${2}/temp
cd ${2}/temp

for i in *
do
	cat $i | tr -s '[:blank:]' ',' > ${2}/fsout/${i%.txt}.csv
done

rm -rf ${1}/temp
rm -rf ${2}/temp
