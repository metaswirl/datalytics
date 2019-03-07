#! /bin/sh
#
# run_plots.sh
# Copyright (C) 2019 Niklas Semmler <niklas.semmler@mailbox.org>
#
# Distributed under terms of the MIT license.
#

for i in $(ls src/plot_*); do
  echo $i
  python3 $i
done
