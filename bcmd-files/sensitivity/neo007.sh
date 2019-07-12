#!/bin/bash

declare -a DATASETS=("neo007" "neo021");
for DATASET in "${DATASETS[@]}"
do
    printf "Working on dataset %s\n" "${DATASET}"
    # DATASET="neo007"
    OUTDIR1=`readlink -m "../../data/SA_results/${DATASET}"`
    OUTDIR2=`readlink -m "../../data/SA_results/${DATASET}_zero"`
    
    WORKDIR=$1
    JOBFILE1="${WORKDIR}${DATASET}_SA.dsimjob"
    JOBFILE2="${WORKDIR}${DATASET}_SA_zero.dsimjob"
    
    DATAFILE="../../data/formatted_data/${DATASET}_formatted.csv"
    
    echo "Writing to ${OUTDIR1}"
    mkdir -p ${OUTDIR1}
    mkdir -p ${OUTDIR2}
    python ~/repos/Github/BayesCMD/batch/dsim.py -o ${OUTDIR1} -b ~/repos/Github/BayesCMD/build ${JOBFILE1} ${DATAFILE}
    
    echo "Writing to ${OUTDIR2}"
    mkdir -p ${OUTDIR2}
    mkdir -p ${OUTDIR2}
    python ~/repos/Github/BayesCMD/batch/dsim.py -o ${OUTDIR2} -b ~/repos/Github/BayesCMD/build ${JOBFILE2} ${DATAFILE}
done

