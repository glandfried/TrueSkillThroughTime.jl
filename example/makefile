all: data synthetic longtest

synthetic: ./output/smoothing.csv
longtest: ./output/longtest_output.csv

./output/smoothing.csv:
	julia synthetic.jl
./output/longtest_output.csv:
	julia longtest.jl

data:
	make -C input




