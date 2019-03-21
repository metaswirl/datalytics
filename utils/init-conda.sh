#! /bin/sh
#
# init-conda.sh
# Copyright (C) 2019 Niklas Semmler <niklas.semmler@mailbox.org>
#
# Distributed under terms of the MIT license.
#

[ -f ~/miniconda3/etc/profile.d/conda.sh ] && source ~/miniconda3/etc/profile.d/conda.sh
[ -f ~/anaconda3/etc/profile.d/conda.sh ] && source ~/anaconda3/etc/profile.d/conda.sh
conda create --file datalytics/structure/configs/conda-spec-file.txt -n $(basename "$PWD")



