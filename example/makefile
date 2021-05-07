all: data 
	julia example.jl

atp: data
	julia atp.jl
	
synthetic: ./output/smoothing.csv
longtest: ./output/longtest_output.csv

./output/smoothing.csv:
	julia synthetic.jl
./output/longtest_output.csv:
	julia ogs.jl

data:
	make -C input




