#!/bin/bash

for i in $(seq -f "%06g" 1 20); do
  python -m megapose.scripts.convert_cnos_to_bop \
    --input local_data/bop23_default_detections_for_task4/cnos-fastsam/cnos-fastsam_tless-test_8ca61cb0-4472-4f11-bce7-1362a12d396f.json \
    --output local_data/bop_datasets/tless/test_primesense/${i} \
    --threshold 0.65
done
