#!/bin/bash

# 確保輸出目錄存在
mkdir -p ./output

# Infinity 測試
# python hps_inference.py /media/ee303/4TB/Infinity/t2i_compare/200_PP --out ./output/infinity_200PP.csv --device cuda --batch-size 4
python hps_inference.py /media/ee303/4TB/Infinity/t2i_compare/50_GP --out ./output/infinity_50GP.csv --device cuda --batch-size 8

# Flux2 測試
python hps_inference.py /media/ee303/4TB/flux2/prompt_test_512/200_PP --out ./output/flux2_200PP.csv --device cuda --batch-size 8
python hps_inference.py /media/ee303/4TB/flux2/prompt_test_512/50_GP --out ./output/flux2_50GP.csv --device cuda --batch-size 8

# Sana 測試
python hps_inference.py /media/ee303/4TB/Sana/t2i_compare/200_PP --out ./output/Sana_200PP.csv --device cuda --batch-size 8
python hps_inference.py /media/ee303/4TB/Sana/t2i_compare/50_GP --out ./output/Sana_50GP.csv --device cuda --batch-size 8