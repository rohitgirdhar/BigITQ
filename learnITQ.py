from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql import SQLContext
from pyspark.ml.feature import PCA
from pyspark.mllib.stat import Statistics
import numpy as np
import sys
import time

# constants
features_path = sys.argv[1]
NDIM = int(sys.argv[2])
NBITS = int(sys.argv[3])
NITER = int(sys.argv[4])
PCA_OUT_PATH = sys.argv[5]
ROT_OUT_PATH = sys.argv[6]
MEAN_OUT_PATH = sys.argv[7]
COMPUTE_PCA = int(sys.argv[8])
PERC_KEEP = float(sys.argv[9])  # a ratio of 1.0 of the number of features to use for learning ITQ

conf = SparkConf().setAppName('BigITQ').set('spark.network.timeout', '500s')  # this fixed the timeout issues with large num of nodes
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
np.random.seed(1)  # only for reproducability


def parse_line(line, perc_keep=1.0, invalid_feat=None):
    if len(line.strip()) == 0:
      return invalid_feat
    if np.random.rand() < perc_keep:
        try:
            feature = (Vectors.dense([float(x) for x in line.split(',')]),)
            return feature
        except Exception, e:
            return invalid_feat
    else:
        return invalid_feat


def learn_pca_embedding(raw_data_frame):
    pca_computer = PCA(k=NBITS, inputCol='features', outputCol='pca')
    pca_model = pca_computer.fit(raw_data_frame)
    return pca_model


def pca_embed(raw_data_frame, pca_model):
    return pca_model.transform(raw_data_frame)


def save_pca_parameters(pca_model, data_dim):
    # since there's no good way of doing it in python, simply use an I matrix to retrieve
    features = [(Vectors.dense(x),) for x in np.eye(data_dim).tolist()]
    params = pca_embed(sqlContext.createDataFrame(features, ('features',)), pca_model)
    np.savetxt(PCA_OUT_PATH,
               np.matrix(params.select('pca').rdd.map(lambda r: r[0]).collect()),
               fmt='%.6f')


def compute_mean(tx_data_rdd):
    summary = Statistics.colStats(tx_data_rdd)
    return summary.mean()


def center_data(raw_data, mean):
    return raw_data.map(lambda x: (Vectors.dense((x[0] - mean).tolist()),))


def mul_ux(z_row, v_row):
    z_row[z_row < 0] = -1
    z_row[z_row >= 0] = 1
    return np.matrix(z_row).transpose().dot(np.matrix(v_row))


def main():
    np.set_printoptions(suppress=True, precision=3)
    data = sc.textFile(features_path).map(lambda x: parse_line(x, PERC_KEEP)).filter(lambda x: x)
    print data.count()
    if COMPUTE_PCA:
        data_mean = compute_mean(data)
        np.savetxt(MEAN_OUT_PATH, data_mean, fmt='%.6f')
        centered_data = center_data(data, data_mean)
        mat_df = sqlContext.createDataFrame(centered_data, ('features',))
        mat_df.cache()
        pca_model = learn_pca_embedding(mat_df)
        # save the PCA embedding
        save_pca_parameters(pca_model, len(data_mean))
        mat_df_tx = pca_embed(mat_df, pca_model)
        mat_df_tx_rdd = mat_df_tx.select('pca').rdd.map(lambda r: r[0])
    else:
        print 'Using the pre-saved PCA parameters and mean'
        data_mean = np.loadtxt(MEAN_OUT_PATH)
        centered_data = center_data(data, data_mean)
        pc = np.loadtxt(PCA_OUT_PATH)
        mat_df_tx_rdd = centered_data.map(lambda x: Vectors.dense(x[0].dot(pc)))
    # print np.matrix(mat_df_tx_rdd.collect())
    centered_data_idx = mat_df_tx_rdd.zipWithIndex().map(lambda (v, k): (k, v))
    centered_data_idx.cache()  # this made it almost 2.8x faster for 60K features 256bit
    # print centered_data.rows.collect()

    # start ITQ Training
    rot_matrix = np.random.randn(NBITS, NBITS)

    u_matrix, _, _ = np.linalg.svd(rot_matrix)
    rot_matrix = u_matrix[:, :NBITS]
    # print rot_matrix

    for iter_id in range(NITER):
        start_time = time.time()
        # print centered_data_idx.collect()
        print('Running iteration %d' % (iter_id + 1))
        # print centered_data_idx.collect()
        z = centered_data_idx.map(lambda (k, row): (k, row.dot(rot_matrix)))
        z.cache()
        # print z.collect()
        c = z.join(centered_data_idx)\
            .map(lambda (k, (z_row, v_row)): (0, mul_ux(z_row, v_row)))\
            .reduceByKey(lambda x, y: x + y).collect()
        # print np.matrix(c[0][1])
        ub, _, ua = np.linalg.svd(c[0][1])
        rot_matrix_old = rot_matrix.copy()
        rot_matrix = np.array(ua.transpose().dot(ub.transpose()))  # IMP: transpose ua bc it's not matlab!
        print('Distance of new matrix: %f' %
              np.linalg.norm(np.matrix(rot_matrix_old) - np.matrix(rot_matrix)))
        print('Time for iteration: %f' % (time.time() - start_time))
    np.savetxt(ROT_OUT_PATH, rot_matrix, fmt='%.6f')
    
if __name__ == '__main__':
    main()
