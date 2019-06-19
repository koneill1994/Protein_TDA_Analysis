#!/bin/bash

DISTFOLD="./DISTFOLD-v0.1/distfold.pl"

maxjobs=0
for rr in `ls ./psicov150-testset/my-rr-predictions/*.rr`;
   do
	id=$(basename $rr)
	id=${id%.*}
	echo $id
    rm -r ./distfold-jobs/output-$id
    # Run 150 jobs in parallel
	perl $DISTFOLD -rr ./psicov150-testset/my-rr-predictions/$id.rr -ss ./psicov150-testset/ss/$id.ss -o ./distfold-jobs/output-$id -mcount 20 -selectrr 1.0L &> ./distfold-jobs/$id.log &
    maxjobs=$((maxjobs+1))
    if [[ "$maxjobs" -gt 10 ]]; then
        exit 0
    fi
   done
