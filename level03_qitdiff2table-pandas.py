#!/usr/bin/env python
import sys
import os
import pandas as pd
from utils.qfsmeas import column_names
df2 = []
for subject in sorted(os.listdir(sys.argv[1])):
 for visit in sorted(os.listdir(sys.argv[1]+"/"+subject+"/qitout")):
  print("loading "+subject+"/"+visit)
  column_names=[]
  column_values=[subject,visit]
  if os.path.isdir(sys.argv[1]+"/"+subject+"/qitout/"+visit+"/tone.fs.map"):
   for j in sorted(os.listdir(sys.argv[1]+"/"+subject+"/qitout/"+visit+"/tone.fs.map")):
    temp=pd.read_csv(sys.argv[1]+"/"+subject+"/qitout/"+visit+"/tone.fs.map/"+j)
    column_names+=["_"+j.replace(".csv", "").replace(".","_")+"_"+k for k in temp["name"]]
    column_values+=list(temp["value"])
  for region in ["ccwm","dkbm","dkgm","dkwm","lbbm","lbgm","lbwm","scgm","wbbm"]:
   if os.path.isdir(sys.argv[1]+"/"+subject+"/qitout/"+visit+"/tone.region/fs."+region+".dti.map"):
    for j in sorted(os.listdir(sys.argv[1]+"/"+subject+"/qitout/"+visit+"/tone.region/fs."+region+".dti.map")):
     temp=pd.read_csv(sys.argv[1]+"/"+subject+"/qitout/"+visit+"/tone.region/fs."+region+".dti.map/"+j)
     column_names+=[region+"_"+j.replace(".csv", "")+"_"+k for k in temp["name"]]
     column_values+=list(temp["value"])
  df2.append(column_values)
pd.DataFrame(df2,columns=column_names).to_csv(sys.argv[2],index=False)
