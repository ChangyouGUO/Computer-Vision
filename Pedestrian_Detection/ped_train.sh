#!/bin/bash

./buildRelease/bin/ped_train -p /home/guo/myDisk/Dataset/INRIAPerson/process_data/train/pos \
    -n /home/guo/myDisk/Dataset/INRIAPerson/process_data/train/neg \
    -r /home/guo/Codes/my-repository/Computer-Vision/Pedestrian_Detection/model/pedestrian_params \
    -m /home/guo/Codes/my-repository/Computer-Vision/Pedestrian_Detection/model/pedestrian_model
