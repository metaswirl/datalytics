#! /bin/sh
#
# create-dirs.sh
# Copyright (C) 2019 Niklas Semmler <niklas.semmler@mailbox.org>
#
# Distributed under terms of the MIT license.
#

mkdir -p data/external data/raw data/processed data/result
touch data/raw/.gitkeep
touch data/external/.gitkeep
touch data/processed/.gitkeep
touch data/result/.gitkeep
touch references/.gitkeep
mkdir -p src/data src/visualization src/result
mkdir -p reports/figures
mkdir -p references
mkdir -p configs
