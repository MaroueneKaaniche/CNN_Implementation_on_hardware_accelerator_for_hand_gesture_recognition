#!/bin/sh
# python train.py --model custommodel2 --dataset CustomDataset2 --confusion --evaluate --device MAX78000 --exp-load-weights-from /home/maroueneubuntu/Desktop/NewMaxProject/Gesture-recognition-synthesis/trained/custom-qatbest8-q.pth.tar -8 --use-bias "$@"
python train_withCrossVal.py --model custommodel2 --dataset CustomDataset2 --confusion --evaluate --device MAX78000 --exp-load-weights-from /home/maroueneubuntu/Desktop/NewMaxProject/Gesture-recognition-synthesis/trained/custom-qat8-q.pth.tar -8 --use-bias --save-sample 13 "$@"
