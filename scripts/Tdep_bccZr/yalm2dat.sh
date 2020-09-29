#!/bin/bash

for i in *.yaml
do
   bandplot --gnuplot $i > $i.dat 
done 

rm *.yaml
