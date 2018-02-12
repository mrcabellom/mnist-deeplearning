import cntk
from mnistdata.mnist_downloader import MnistDownloader
from mnistdata.ctf_reader import init_reader
from settings import MINST_DOWNLOAD_INFO
from batchlog.training import progress
from batchlog.tensor_writer import TensorWriter
from utils.devices import set_devices
from evaluation.test_mnist import predict_minist_images
from models.logistic_regression import LogisticRegression


if __name__ == '__main__':

    set_devices()
    input_dim = 784
    input_dim_model = (1, 28, 28)
    num_output_classes = 10

    mnist_downloader = MnistDownloader(MINST_DOWNLOAD_INFO)
    mnist_downloader.download_and_save_data()

    model_definition = LogisticRegression(
        input_dim, num_output_classes)

    learning_rate = 0.2
    lr_schedule = cntk.learning_rate_schedule(
        learning_rate, cntk.UnitType.minibatch)
    learner = cntk.sgd(model_definition.model.parameters, lr_schedule)
    tensor_writer = TensorWriter(model_definition.model)

    trainer = cntk.Trainer(
        model_definition.model,
        (model_definition.get_loss(), model_definition.get_classification_error()),
        [learner], tensor_writer.get_writer())

    # Trainning
    minibatch_size = 64
    num_samples_per_sweep = 60000
    num_sweeps_to_train_with = 10
    num_minibatches_to_train = (
        num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size

    reader_train = init_reader(
        mnist_downloader.train_file, input_dim, num_output_classes)
    input_map = {
        model_definition.label: reader_train.streams.labels,
        model_definition.input: reader_train.streams.features      
    }

    for i in range(0, int(num_minibatches_to_train)):
        data = reader_train.next_minibatch(minibatch_size, input_map=input_map)
        output = trainer.train_minibatch(
            data, outputs=[model_definition.input])
        tensor_writer.write_model_params(i)
        output[1][model_definition.input] = output[1][model_definition.input].reshape(minibatch_size,1,28,28)       
        tensor_writer.write_image(output[1], i)
        batchsize, loss, error = progress(
            trainer, i, frequency=500, verbose=1)

    # Test
    reader_test = init_reader(
        mnist_downloader.test_file, input_dim, num_output_classes, is_training=False)

    test_input_map = {
        model_definition.label: reader_test.streams.labels,
        model_definition.input: reader_test.streams.features
    }

    test_minibatch_size = 512
    num_samples = 10000
    num_minibatches_to_test = num_samples // test_minibatch_size
    test_result = 0.0

    for i in range(num_minibatches_to_test):
        data = reader_test.next_minibatch(test_minibatch_size,
                                          input_map=test_input_map)
        eval_error = trainer.test_minibatch(data)
        test_result = test_result + eval_error

    print("Average test error: {0:.2f}%".format(
        test_result * 100 / num_minibatches_to_test))

    predict_minist_images(model_definition, mnist_downloader.test_file)

