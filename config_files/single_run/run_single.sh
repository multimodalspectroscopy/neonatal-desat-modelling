DATASET="neo007"
DATAFILE="/home/buck06191/repos/Github/neonatal-desat-modelling/data/formatted_data/${DATASET}_formatted.csv"
CONFIG="/home/buck06191/repos/Github/neonatal-desat-modelling/config_files/single_run/${DATASET}.json"

echo ${DATAFILE}
echo ${CONFIG}
python "/home/buck06191/repos/Github/BayesCMD/scripts/single_run/run_model.py" --workdir "/home/buck06191/repos/Github/neonatal-desat-modelling/data/debug_results/${DATASET}/" ${DATAFILE} ${CONFIG}
