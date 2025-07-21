#!/bin/bash

for i in $(seq -f "%06g" 1 8); do
  rm -rf local_data/bop_datasets/isc/test/${i}/mask_visib \
         local_data/bop_datasets/isc/test/${i}/scene_gt.json \
         local_data/bop_datasets/isc/test/${i}/scene_gt_info.json
done
