#!/bin/bash

#Option=('Show')
Option=('Train')
Runs=('2')

for clf in "${Option[@]}"
do
    for i in "${Runs[@]}"
    do
      # Call the experiment

      python3 run.py $clf $i


    done
done
