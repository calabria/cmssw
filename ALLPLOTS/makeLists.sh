#!/bin/bash

path=$1
root=$(echo `pwd`)
toFind=$2

rm $root/Convert/txt/*.txt
rm $root/Convert/txt/sum/*.txt

for s in `ls -l $root/$path | awk '{print $9}' | grep $toFind`; do

	echo $s

	for k in `ls -l $path/$s | awk '{print $9}' | grep 'log'`; do

		for i in `cat $path/$s/$k/crab.log | grep 'validationEDM' | grep 'lfn'`; do

			echo $i | grep 'store' >> ./Convert/txt/$s'_tot.txt'

		done

	done

done

cd $root/Convert/txt/
./sum.sh
