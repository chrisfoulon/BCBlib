#! /bin/bash
[ $# -lt 1 ] && { echo 'Usage :
  $1 : root directory
  [$2 : prefix of the input file for randomise inside the subfolders]'; exit 1; }


# randomise -i toto_filtered_4D.nii.gz -o toto_filtered_4D.nii.gz -d toto_design.mat -t design.con -T -n 10000


for d in "$1"*/;
do
  cd "$d" || echo "Could not change directory in $d"; continue
#  test if the files exist
  echo "$d";
done;