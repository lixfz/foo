#!/bin/bash

REPORT_DIR=${1:-.}

if [ "$2" == "ttft" ]; then
  op=".streaming_metrics.avg_ttft_ms"
else
  op=".performance.system_output_throughput_tok_s"
fi

echo fetch: $op
echo ""

for f in `ls $REPORT_DIR/*.json`; do
  #r=`jq .performance.system_output_throughput_tok_s  $f`
  r=`jq $op $f`
  echo $f: $r
done
