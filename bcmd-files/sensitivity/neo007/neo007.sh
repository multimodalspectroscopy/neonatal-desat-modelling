#!/bin/bash

DATASET="neo007"
OUTDIR1=`readlink -m "../../data/SA_results/bph_2/with_cellDeath/nrmse/${DATASET}"`
OUTDIR2=`readlink -m "../../data/SA_results/bph_2/with_cellDeath/nrmse_zero/${DATASET}"`

WORKDIR=$1
JOBFILE1="${WORKDIR}${DATASET}_SA.dsimjob"
JOBFILE2="${WORKDIR}${DATASET}_SA_zero.dsimjob"

DATAFILE="../../data/formatted_desat/${DATASET}_filtered_formatted.csv"

echo "Writing to ${OUTDIR1}"
mkdir -p ${OUTDIR1}
mkdir -p ${OUTDIR2}
python ~/repos/Github/BayesCMD/batch/dsim.py -o ${OUTDIR1} -b ~/repos/Github/BayesCMD/build ${JOBFILE1} ${DATAFILE}

echo "Writing to ${OUTDIR2}"
mkdir -p ${OUTDIR2}
mkdir -p ${OUTDIR2}
python ~/repos/Github/BayesCMD/batch/dsim.py -o ${OUTDIR2} -b ~/repos/Github/BayesCMD/build ${JOBFILE2} ${DATAFILE}

