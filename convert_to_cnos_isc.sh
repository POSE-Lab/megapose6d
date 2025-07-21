#!/bin/bash

for i in $(seq -f "%06g" 1 8); do
  python -m megapose.scripts.convert_cnos_to_bop \
    --input local_data/bop23_default_detections_for_task4/cnos-fastsam/isc.json \
    --output local_data/bop_datasets/isc/test/${i} \
    --threshold 0.0
done
