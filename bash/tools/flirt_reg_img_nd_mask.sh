#! /bin/bash

source ~/.profile
[ $# -lt 2 ] && { echo 'Usage :
  $1 = the image to register
  $2 = reg image output folder
  $3 = template (for instance MNI152 1mm)
  $4 = the mask to register
  $5 = reg mask output folder'; exit 1; }

flirt -in $1 -ref $3 -out $2/reg_$(basename $1) -omat $2/reg_$(basename $1).mat

flirt -in $4 -ref $3 -interp nearestneighbour -applyxfm -init $2/reg_$(basename $1).mat -out $5/reg_$(basename $4)

