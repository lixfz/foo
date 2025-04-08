#!/bin/bash


for i in `seq 1 1000`
do
	# ls -l flops_profiler_$i.txt
	[ -f flops_profiler_$i.txt ] && grep '^FLOPS per GPU' flops_profiler_$i.txt
	# [ -f flops_profiler_$i.txt ] && grep '^fwd MACs per GPU' flops_profiler_$i.txt
done | tail -n 20

date
ls -lrt | tail -n1
