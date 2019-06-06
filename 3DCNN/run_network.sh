#!/bin/bash

###################
# TRAINING CONFIG #
###################

export ROOT= "/root/3DCNN"
export LABELS = "dummy.json"
export IS_CROPPED="--is_cropped"  # Uncomment to crop input images
export CROP_SIZE="112 112"
export BATCHSIZE="128"
export VAL_BATCHSIZE = "4"
export FRAMES="16"
export SHUFFLE="--shuffle"
export FCN="--FCN"
export NORMALIZE="--normalized"
export RANDOM_FLIP="--random_flip"
export COLOR_SPACE="RGB"
export DIM_ORDER="cfhw"
export LR="0.01"
export MOMENTUM="0.9"
export WEIGHT_DECAY="1e-4"
export STEP_SIZE="5"
export GAMMA="0.1"
export EPOCHS="90"
export ARCH="c3d"
export NUM_CLASSES="2"
export OUTPUT="./results/`date +%b%d_%H-%M-%S`"
export PRINTFREQ="500"
export STRIDE="8"

echo $OUTPUT

tensorboard --logdir runs 2> /dev/null &# $OUTPUT 2> /dev/null &
echo "Tensorboard launched"

python main.py --labels_json $LABELS --batchsize $BATCHSIZE -- val_batchsize $VAL_BATCHSIZE --frames $FRAMES $IS_CROPPED --crop_size $CROP_SIZE $SHUFFLE $NORMALIZE $RANDOM_FLIP --color_space $COLOR_SPACE --dimension_order $DIM_ORDER --lr $LR --momentum $MOMENTUM --epochs $EPOCHS --arch $ARCH --num_classes $NUM_CLASSES --weight-decay $WEIGHT_DECAY --output $OUTPUT $FCN $SINGLE_GPU --print-freq $PRINTFREQ --stride $STRIDE --step_size $STEP_SIZE --gamma $GAMMA





###################
# TESTING CONFIG #
###################

export TEST = "--test"
export IS_CROPPED="--is_cropped"  # Uncomment to crop input images
export CROP_SIZE="112 112"
export BATCHSIZE="128"
export VAL_BATCHSIZE = "4"
export FRAMES="16"
export SHUFFLE="--shuffle"
export FCN= # "--FCN"
export NORMALIZE="--normalized"
export RANDOM_FLIP="--random_flip"
export COLOR_SPACE="RGB"
export DIM_ORDER="cfhw"
export LR="0.01"
export MOMENTUM="0.9"
export WEIGHT_DECAY="1e-4"
export STEP_SIZE="5"
export GAMMA="0.1"
export EPOCHS="90"
export ARCH="c3d"
export NUM_CLASSES="2"
export OUTPUT="./test_results/"
export PRINTFREQ="100"
export STRIDE="8"
export RESUME="./results/weights/model_best.pth.tar"

echo $OUTPUT

python main.py $TEST --batchsize $BATCHSIZE -- val_batchsize $VAL_BATCHSIZE --frames $FRAMES $IS_CROPPED --crop_size $CROP_SIZE $SHUFFLE $NORMALIZE $RANDOM_FLIP --color_space $COLOR_SPACE --dimension_order $DIM_ORDER --lr $LR --momentum $MOMENTUM --epochs $EPOCHS --arch $ARCH --num_classes $NUM_CLASSES --weight-decay $WEIGHT_DECAY --output $OUTPUT $FCN $SINGLE_GPU --print-freq $PRINTFREQ --stride $STRIDE --step_size $STEP_SIZE --gamma $GAMMA --resume $RESUME