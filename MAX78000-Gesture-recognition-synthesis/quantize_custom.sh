#!/bin/sh
python quantize.py /home/maroueneubuntu/Desktop/NewMaxProject/Gesture-recognition-training/logs/2023.04.18-091402/qat_qat_qat_qat_qat_best.pth.tar trained/custom-qatbest8-q.pth.tar --device MAX78000 -v "$@"
# python quantize.py /home/maroueneubuntu/Desktop/NewMaxProject/Gesture-recognition-training/logs/2023.04.18-091402/qat_qat_qat_qat_qat_checkpoint.pth.tar trained/custom-qat8-q.pth.tar --device MAX78000 -v --scale 0.85 "$@"
