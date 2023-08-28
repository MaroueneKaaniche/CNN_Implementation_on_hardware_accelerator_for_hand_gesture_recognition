#!/bin/sh
python train_withCrossVal.py --lr 0.001 --optimizer Adam --epochs 250 --model custommodel2 --tensorboard --qat-policy policies/qat_policy.yaml --validation-split 0 --dataset CustomDataset2 --data home/maroueneubuntu/Desktop/CNN/ --batch-size 32 --device MAX78000 --pr-curves "$@"
# --confusion --show-train-accuracy full --compress policies/schedule_custom.yaml --tensorboard --validation-split 0