#!/bin/sh
./evaluate2.py --onnx-file export/customdataloader_tf/saved_model.onnx --dataset customdataloader_tf
./evaluate2.py --onnx-file export/customdataloader_tf/saved_model_dq.onnx --dataset customdataloader_tf