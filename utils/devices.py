from cntk import device
import cntk as cn


def set_devices():
    if (cn.__version__ != '2.2'):
        raise Exception('Invalid CNTK Version')
    all_devices = device.all_devices()
    if all_devices[0].type() == device.DeviceKind.GPU:
        print('You can use the GPU of your computer!!!')
        device.try_set_default_device(device.gpu(0))
    else:
        print('Sorry, your computer only has a slow CPU')
        device.try_set_default_device(device.cpu())
