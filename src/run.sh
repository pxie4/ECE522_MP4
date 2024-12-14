#!/bin/bash
echo "Running simulation with BERT"
./sim configs/cxl2/bert.config
if [ $? -ne 0 ]; then
    echo "Error occurred while running run1.config. Exiting."
    exit 1
fi

echo "Running simulation with INCPETION"
./sim configs/cxl2/inception.config
if [ $? -ne 0 ]; then
    echo "Error occurred while running run2.config. Exiting."
    exit 1
fi

echo "Running simulation with RESNET"
./sim configs/cxl2/resnet.config
if [ $? -ne 0 ]; then
    echo "Error occurred while running run1.config. Exiting."
    exit 1
fi

echo "Running simulation with SENET"
./sim configs/cxl2/senet.config
if [ $? -ne 0 ]; then
    echo "Error occurred while running run2.config. Exiting."
    exit 1
fi

echo "Running simulation with VIT"
./sim configs/cxl2/VIT.config
if [ $? -ne 0 ]; then
    echo "Error occurred while running run3.config. Exiting."
    exit 1
fi


echo "All simulations completed successfully!"
