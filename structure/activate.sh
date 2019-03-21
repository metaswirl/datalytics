#! /bin/sh
#
# activate-conda.sh
# Copyright (C) 2019 Niklas Semmler <niklas.semmler@mailbox.org>
#
# Distributed under terms of the MIT license.
#
NAME="$(basename "$PWD")"

echo "Make sure to source and not call this file!"
export PYTHONPATH="$PWD:$PYTHONPATH"
[ -f ~/miniconda3/etc/profile.d/conda.sh ] && source ~/miniconda3/etc/profile.d/conda.sh
[ -f ~/anaconda3/etc/profile.d/conda.sh ] && source ~/anaconda3/etc/profile.d/conda.sh
conda info --envs | tail -n+3 | head -n-1 | cut -f1 -d' ' | grep $NAME &> /dev/null
if [ $? -eq 0 ]; then
  echo "activating conda environment"
  conda activate $(basename "$PWD")
else
  echo "environment $NAME not found"
  echo "creating environment in 3 seconds"
  sleep 3
  conda create --file configs/conda-spec-file.txt -n $(basename "$PWD")
fi
