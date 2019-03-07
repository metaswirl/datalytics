#
# Makefile
# Niklas Semmler, 2019-03-07 20:46
#

PFTYPE := $(if $(PFTYPE),$(PFTYPE),"svg")
DFTYPE=pickle
DATA=$(patsubst src/%.py,%.${DFTYPE}, $(filter-out src/data/common.py, $(wildcard src/data/*.py)))
PLOTS=$(patsubst src/%.py,%.${PFTYPE}, $(filter-out src/plot/common.py, $(wildcard src/plot/*.py)))
RESULTS=$(patsubst src/%.py,%.csv, $(filter-out src/result/common.py, $(wildcard src/result/*.py)))

all: plot

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

#data/clean_select_data.${DFTYPE}: data/select_data.${DFTYPE}
#	python3 src/data/clean_select_data.py
#
#plot/cdf_total_result_record_count.${PFTYPE}: data/clean_select_data.${DFTYPE}
#	python3 src/plot/cdf_total_result_record_count.py
