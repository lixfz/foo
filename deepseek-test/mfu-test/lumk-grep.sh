#!/bin/bash


for i in `seq 1 100`
do
	# ls -l flops_profiler_$i.txt
	[ -f flops_profiler_$i.txt ] && grep '^fwd flops per GPU' flops_profiler_$i.txt
done | tail -n 100

date
ls -lrt | tail -n1
