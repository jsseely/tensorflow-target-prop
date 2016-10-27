#!/bin/bash
#PBS -N targprop
#PBS -W group_list=yetizmbbi
#PBS -l walltime=12:00:00,mem=2000mb
#PBS -M jss2219@cumc.columbia.edu
#PBS -m n
#PBS -V
#PBS -t 0-1000
#PBS -o /vega/zmbbi/users/jss2219/targprop/output/
#PBS -e /vega/zmbbi/users/jss2219/targprop/error/

# Reminder: use source activate tensorflow before running:
# source /vega/zmbbi/users/jss2219/miniconda2/bin/activate tensorflow

# Yeti doesn't have glibc, so have to run it this way:
LD_LIBRARY_PATH="/vega/zmbbi/users/jss2219/glibc/lib/x86_64-linux-gnu/:/vega/zmbbi/users/jss2219/glibc/usr/lib64/" /vega/zmbbi/users/jss2219/glibc/lib/x86_64-linux-gnu/ld-2.17.so `which python` targprop_wrapper.py $PBS_ARRAYID "BBB"
