#!/bin/bash

# script to extract Imagenette and Imagewoof dataset
# https://github.com/fastai/imagenette

wget -c https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
tar -xzf imagenette2.tgz

wget -c https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz
tar -xzf imagewoof2.tgz

rm *.tgz

