#!/bin/bash

# for model in "twoLNN" "fiveLNN" "LeNet" 
# do
# for k in 1 2 3 4 5
# do 
# echo $model $k "parity"
# python train.py -model $model -task "parity" -k $k -lr 0.05
# done
# done

for model in "LeNet" #"twoLNN" "fiveLNN" "LeNet" 
do
for k in 1 2 3 4 5
do 
echo $model $k "classification"
python train.py -model $model -task "classification" -k $k -lr 0.01
done
done

for model in "twoLNN" "fiveLNN" "LeNet" 
do
for k in 3
do 
echo $model $k "classification from scratch"
python train.py -model $model -task "classification" -k $k -lr 0.01 -scratch True
done
done