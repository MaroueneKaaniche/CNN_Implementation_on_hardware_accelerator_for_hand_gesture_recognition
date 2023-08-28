#!/bin/sh
./train_withcrossval.py --epochs 250 --optimizer Adam --lr 0.001  --model custom_model --dataset customdataloader_tf --save-sample 12
./convert.py --saved-model export/customdataloader_tf --opset 10 --output export/customdataloader_tf/saved_model.onnx