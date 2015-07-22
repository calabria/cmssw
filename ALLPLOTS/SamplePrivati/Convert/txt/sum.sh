#!/bin/bash

for s in "5" "10" "50" "100" "500" "1000"; do

	for k in "Official"; do

		name1="MuMinus"$s"_"$k"_tot.txt"
		name2="MuPlus"$s"_"$k"_tot.txt"
		name3="MuMinus"$s"_"$k"_tot.txt"

		cat $name1 $name2 > ./sum/$name3

	done

done

dyname=`ls -l | awk '{print $9}' | grep 'DY'`

echo $dyname
cp $dyname ./sum/
