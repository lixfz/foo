#!/bin/bash

REPORT_DIR=${1:-.}

for f in `ls $REPORT_DIR/*.json`; do
   r=`jq .performance.system_output_throughput_tok_s  $f`
   echo $f: $r
done
