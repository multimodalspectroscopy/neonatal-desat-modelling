#!/bin/bash -l
#$ -l h_rt=1:00:00
#$ -N neo007
#$ -o /home/ucbpjru/Scratch/neo007/out
#$ -e /home/ucbpjru/Scratch/neo007/err
#$ -wd /home/ucbpjru/Scratch
# Set up the job array.  In this instance we have requested 1000 tasks
# numbered 1 to 1000.
#$ -t 20001-40000

module load python3/recommended
cd $TMPDIR
export BASEDIR="${HOME}/BayesCMD"

DATASET="neo007"
DATAFILE="${HOME}/neonatal-desat-modelling/data/formatted_data/${DATASET}_formatted.csv"
CONFIGFILE="${HOME}/neonatal-desat-modelling/config_files/abc/neo_config.json"

echo "Datafile is ${DATAFILE}\nConfig file is ${CONFIGFILE}."

start=`date +%s`
python3 $BASEDIR/scripts/batch.py 1000 $DATAFILE $CONFIGFILE --workdir $TMPDIR
echo "Duration: $(($(date +%s)-$start))" > $TMPDIR/$SGE_TASK_ID.timings.txt

tar -zcvf $HOME/Scratch/${DATASET}/batch_$JOB_NAME.$SGE_TASK_ID.tar.gz $TMPDIR
