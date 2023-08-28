#!/bin/sh
DEVICE="MAX78000"
TARGET="sdk/Examples/$DEVICE/CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose --no-version-check "

python ai8xize.py --test-dir $TARGET --prefix customModel --checkpoint-file trained/custom-qat8-q.pth.tar --config-file networks/custom.yaml --sample-input /home/maroueneubuntu/Desktop/NewMaxProject/Gesture-recognition-training/sample_customdataset2.npy --softmax --overwrite --embedded-code --compact-data --mexpress  $COMMON_ARGS "$@"
