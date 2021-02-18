#!/bin/bash


INPUT_TXT_FILE=~/Dokumenty/audio-codecs/AMR_channel_testing_v6/FILE2.txt
BASE_NAME=mytest;
DIR_PATH=/tmp/test
ENCODER_PATH=./encoder_VAD1
ENCODER_OPT="-dtx MR122"
DECODER_PATH=./decoder_VAD1
mkdir -p $DIR_PATH
[ $? -eq 0 ] || exit 1
cat $INPUT_TXT_FILE | IVS -n 1 >$DIR_PATH/$BASE_NAME.inp

$ENCODER_PATH $ENCODER_OPT $DIR_PATH/$BASE_NAME.inp $DIR_PATH/$BASE_NAME.cod && $DECODER_PATH  $DIR_PATH/$BASE_NAME.cod $DIR_PATH/$BASE_NAME.out

[ $? -eq 0 ] || exit 1
cat $DIR_PATH/$BASE_NAME.out | PSAP -n 1 >$DIR_PATH/log.txt
diff -Naur $INPUT_TXT_FILE $DIR_PATH/log.txt
