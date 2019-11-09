#!/bin/bash -l
#$ -l h_rt=00:10:00
#$ -N gradient_neo007_debug
#$ -o /home/ucbpjru/Scratch/neo007_gradient_debug/out
#$ -e /home/ucbpjru/Scratch/neo007_gradient_debug/err
#$ -wd /home/ucbpjru/Scratch/

module load python3/recommended
#Local2Scratch

# cd $TMPDIR
export BASEDIR="${HOME}/BayesCMD"
DATASET="neo007"
DATAFILE="${HOME}/neonatal-desat-modelling/data/formatted_data/${DATASET}_formatted.csv"
CONFIG="${HOME}/neonatal-desat-modelling/config_files/single_run/${DATASET}.json"

echo ${DATAFILE}
echo ${CONFIG}
python "${HOME}/BayesCMD/scripts/single_run/run_model.py" --workdir . ${DATAFILE} ${CONFIG}

#tar -zcvf $HOME/Scratch/${DATASET}_gradient_debug/batch_$JOB_NAME.update.tar.gz $TMPDIR
