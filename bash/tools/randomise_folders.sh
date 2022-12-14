#! /bin/bash
[ $# -lt 1 ] && { echo 'Usage :
  $1 : root directory
  $2 : number of permutations (default 10000)
  [$3 : prefix of the input file for randomise inside the subfolders]'; exit 1; }

dir=$1

if [[ ! $dir =~ /$ ]];
then
  dir="$dir/"
fi

# randomise -i toto_filtered_4D.nii.gz -o toto_filtered_4D.nii.gz -d toto_design.mat -t design.con -T -n 10000
echo "Starting $dir"
for d in "$dir"*/;
do
  cd "$d" || echo "Could not change directory in $d"
#  test if the files exist
  if [ -e "$d$3"filtered_4D.nii.gz ] && [ -e "$d$3"design.mat ] && [ -e "$d$3"design.con ];
  then
    echo "Running randomise in $d"
    randomise -i "$d$3"filtered_4D.nii.gz -o "$d$3"randomised_4D.nii.gz -d "$d$3"design.mat -t "$d$3"design.con -T -n "$2"
  else
    echo "missing files in $d"
  fi
done;