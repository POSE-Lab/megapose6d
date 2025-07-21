#!/bin/bash

for i in $(seq -f "%06g" 1 26); do
  rm -rf local_data/bop_datasets/tless/test_primesense/${i}/mask_visib \
         local_data/bop_datasets/tless/test_primesense/${i}/scene_gt.json \
         local_data/bop_datasets/tless/test_primesense/${i}/scene_gt_info.json
done
