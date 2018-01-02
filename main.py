import cntk
from mnistdata.mnist_downloader import MnistDownloader
from mnistdata.ctf_reader import init_reader
from settings import MINST_DOWNLOAD_INFO
from utils.devices import set_devices


if __name__ == '__main__':

    set_devices()
    input_dim = 784
    input_dim_model = (1, 28, 28)
    num_output_classes = 10

    mnist_downloader = MnistDownloader(MINST_DOWNLOAD_INFO)
    mnist_downloader.download_and_save_data()
