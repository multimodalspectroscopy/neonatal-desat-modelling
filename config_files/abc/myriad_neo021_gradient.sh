#!/bin/bash -l
#$ -l h_rt=1:00:00
#$ -N gradient_neo021
#$ -o /home/ucbpjru/Scratch/neo021_gradient/out
#$ -e /home/ucbpjru/Scratch/neo021_gradient/err
#$ -wd /home/ucbpjru/Scratch
# Set up the job array.  In this instance we have requested 1000 tasks
# numbered 1 to 1000.
#$ -t 1-40000

module load python3/recommended
cd $TMPDIR
export BASEDIR="${HOME}/BayesCMD"

DATASET="neo021"
DATAFILE="${HOME}/neonatal-desat-modelling/data/formatted_data/${DATASET}_formatted.csv"
CONFIGFILE="${HOME}/neonatal-desat-modelling/config_files/abc/neo_config_gradient.json"

echo "Datafile is ${DATAFILE}\nConfig file is ${CONFIGFILE}."

start=`date +%s`
python3 $BASEDIR/scripts/batch.py 1000 $DATAFILE $CONFIGFILE --workdir $TMPDIR
echo "Duration: $(($(date +%s)-$start))" > $TMPDIR/$SGE_TASK_ID.timings.txt

tar -zcvf $HOME/Scratch/${DATASET}_gradient/batch_$JOB_NAME.$SGE_TASK_ID.tar.gz $TMPDIR
