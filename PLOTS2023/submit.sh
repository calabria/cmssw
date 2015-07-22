#!/bin/bash

for i in `ls -l | awk '{print $9}' | grep 'Case'`; do

	echo $i
	#crab -submit -c $i
	#crab -kill all -c $i
	crab -status -c $i | grep '60307'
	#crab -get -c $i


done
