# DSET=Holidays1M
DSET=CIFAR10

FEAT=Work/BigITQ/${DSET}/fc7.csv
PRECOMP_PCA=../Expts/precomp/${DSET}/PCA.txt
PRECOMP_MEAN=../Expts/precomp/${DSET}/Mean.txt
/usr/lib/spark/bin/spark-submit \
    --master yarn-client \
    --driver-memory 4g \
    --executor-memory 150g \
    --executor-cores 15 \
    --num-executors 10 \
    learnITQ.py \
      $FEAT \
      4096 \
      256 \
      50 \
      $PRECOMP_PCA \
      scratch/R.txt \
      $PRECOMP_MEAN \
      1 \
      1.0 \
    2>&1 | tee stdout.txt
