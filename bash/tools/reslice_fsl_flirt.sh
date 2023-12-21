#! /bin/bash
[ $# -lt 2 ] && { echo 'Usage :
  $1 = the image to be converted
  $2 = template (for instance MNI152 1mm)
  $3 = matrix without tranformation
  $4 = resultFolder'; exit 1; }

flirt -in $1 -ref $2 -applyxfm -init $3 \
  -out $4/$(basename $1)
