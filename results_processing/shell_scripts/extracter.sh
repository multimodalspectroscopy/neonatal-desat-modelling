#!/bin/bash
declare -a neonates=("neo007" "neo021")

for N in ${neonates[@]};
do
    printf "Working on neonate %s\n" "${N}"
    for i in {1..40000};
    do
        printf "\r\tWorking on batch %6d\n" "${i}"
        OUTPUT_DIR="/home/buck06191/Dropbox/phd/desat_neonate/ABC/gradient_SA/${N}/params_$i"
        ARCHIVES_DIR="/home/buck06191/Legion_Archives/desat_neonate/abc/gradient/${N}_gradient/batch_gradient_${N}.$i.tar.gz"
        mkdir -p $OUTPUT_DIR
        tar -xzf $ARCHIVES_DIR -C $OUTPUT_DIR --wildcards --no-anchored --strip-components 3 '*/*/*/parameters.csv'
    done
done
