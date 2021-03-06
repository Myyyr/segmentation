from .memory_unet_3D import *
from .revunet_3D import *
from .unet_3D import *

def get_network(name, n_classes, in_channels=3, feature_scale=4, tensor_dim='2D',
                nonlocal_mode='embedded_gaussian', attention_dsample=(2,2,2),
                aggregation_mode='concat', im_dim=None):
    print('seg model _get_model_instance ...')
    model = _get_model_instance(name, tensor_dim)
    print('seg model _get_model_instance done ...')

    
    if name in ['unet_3d']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      feature_scale=feature_scale,
                      is_deconv=False,
                      im_dim = im_dim)

    elif name in ['memory_unet_3d']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      feature_scale=feature_scale,
                      is_deconv=False,
                      im_dim = im_dim)


    elif name in ['revunet']:
        model = model()
    
    else:
        raise 'Model {} not available'.format(name)

    return model


def _get_model_instance(name, tensor_dim):
    return {
        'unet_3d' : {'3D':unet_3D},
        'revunet' : {'3D':RevUnet3D},
        'memory_unet_3D' : {'3D':memory_unet_3D}
    }[name][tensor_dim]


