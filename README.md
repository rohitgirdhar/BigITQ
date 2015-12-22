# BigITQ
Distributed implementation of Iterative Quantization<sup>1</sup> over SPARK

A report on this work is avaiable now: [pdf](https://github.com/rohitgirdhar/BigITQ/raw/master/report.pdf)

## Running the Code

The python script is designed to run over spark. However, it can always be run on a single node also with spark
installed. However, since the implementation is not vectorized, it is advisable to use the original or alternate
implementations for a single machine. `run.sh` is an example run script I used.

```bash
# note that all the paths below are in HDFS
FEAT_PATH=/path/to/features.csv  # Path to a csv file with each row as a feature vector
PCA_PATH=/path/to/PCA.txt  # Path to store or load the PCA projection matrix (CSV with delimiter=space)
MEAN_PATH=/path/to/Mean.txt  # Path to store or load the Mean feature from the data (CSV with delimiter=newline)
R_PATH=/path/to/R.txt  # Path to store the output learnt rotation matrix
$ /usr/lib/spark/bin/spark-submit \
    --master yarn-client \
    --driver-memory 4g \
    --executor-memory 150g \
    --executor-cores 15 \
    --num-executors 10 \
    learnITQ.py \
      $FEAT_PATH \
      4096 \  # feature dimension, this is for CNN fc7
      256 \  # output binary hash dimension
      50 \  # number of iterations to run
      $PCA_PATH \
      $R_PATH \
      $MEAN_PATH \
      1 \  # set =1 to recompute PCA and Mean, else load from the files
      1.0 \  # amount of the data to use (between 0.0 to 1.0, 1.0 means all the data)
    2>&1 | tee stdout.txt  # store the output of this run into stdout.txt 
```


<sup>1</sup>Yunchao Gong, S. Lazebnik, A. Gordo, and F. Perronnin. Iterative Quantization: A Procrustean Approach to Learning Binary Codes for Large-scale Image Retrieval. Accepted, IEEE Transaction on Pattern Analysis and Machine Intelligence, (TPAMI), 2012
