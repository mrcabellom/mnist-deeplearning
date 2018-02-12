DATA_FOLDER = 'mnistdata/files/'
MINST_DOWNLOAD_INFO = {
    'BASE_URL': 'http://yann.lecun.com/exdb/mnist/',
    'TRAIN_DATA': {
        'NAME_IMAGE': 'train-images-idx3-ubyte.gz',
        'NAME_LABELS': 'train-labels-idx1-ubyte.gz',
        'SAMPLES': 60000
    },
    'TEST_DATA': {
        'NAME_IMAGE': 't10k-images-idx3-ubyte.gz',
        'NAME_LABELS': 't10k-labels-idx1-ubyte.gz',
        'SAMPLES': 10000
    }
}
TENSOR_LOG_DIR = 'log'
