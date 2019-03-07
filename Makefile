#
# Makefile
# Niklas Semmler, 2019-03-07 20:46
#

DATA=$(patsubst src/%.py,%.pickle, $(filter-out src/data/common.py, $(wildcard src/data/*.py)))
PLOTS=$(patsubst src/%.py,%.svg, $(filter-out src/plot/common.py, $(wildcard src/plot/*.py)))
RESULTS=$(patsubst src/%.py,%.csv, $(filter-out src/result/common.py, $(wildcard src/result/*.py)))

all: data plot

data: ${DATA}

plot: ${PLOTS}

init:
	mkdir -p data/
	mkdir -p input/
	mkdir -p src/data src/plot src/result
	mkdir -p plot
	mkdir -p result

clean:
	rm -f data/*
	rm -f plot/*

#plot/bar_queries_by_user.svg: data/select_data.pickle
#	python3 src/plot/bar_queries_by_user.py

#data/clean_select_data.pickle: data/select_data.pickle
#	python3 src/data/clean_select_data.py
