#!/usr/bin/env python
import sys
import os
import pandas as pd
df2 = []
for subject in sorted(os.listdir(sys.argv[1])):
 for visit in sorted(os.listdir(sys.argv[1]+"/"+subject+"/qitout")):
  print("loading "+subject+"/"+visit)
  for modality in ["dkgm","dkwm","scgm"]:
    csv_path=sys.argv[1]+"/"+subject+"/qitout/"+visit+"/tone.region/fs."+modality+".dti.map"
    if os.path.isdir(csv_path):
     column_names=[]
     column_values=[subject,visit]
     for j in sorted(os.listdir(csv_path)):
      temp=pd.read_csv(csv_path+"/"+j)
      column_names=[modality+"_"+j.replace(".csv", "")+"_"+k for k in temp["name"]]
      column_values+=list(temp["value"])
      df2.append(column_values)
pd.DataFrame(df2,columns=["Subject","Visit"]+column_names).to_csv(sys.argv[2]+"/qitout.all.csv",index=False)
