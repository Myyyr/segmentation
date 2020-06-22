from .unet_2D import *
from .unet_3D import *
from .unet_nonlocal_2D import *
from .unet_nonlocal_3D import *
from .unet_grid_attention_3D import *
from .unet_CT_dsv_3D import *
from .unet_CT_single_att_dsv_3D import *
from .unet_CT_multi_att_dsv_3D import *
from .sononet import *
from .sononet_grid_attention import *
from .revunet_3D import *
from .revunet_3D_big import *
from .vnet import *
from .revunet_3D_dsv import *
from .revunet_3D_deep1_dsv import *
from .revunet_3D_wide_dsv import *
from .fully_reversible import *
from .iunet import *
from .iunet_layers import create_double_module

def get_network(name, n_classes, in_channels=3, feature_scale=4, tensor_dim='2D',
                nonlocal_mode='embedded_gaussian', attention_dsample=(2,2,2),
                aggregation_mode='concat'):
    model = _get_model_instance(name, tensor_dim)

    if name in ['revunet', 'revunet_big', 'revunet_dsv', 'revunet_deep_dsv', "revunet_wide_dsv", "fully_reversible"]:
        model = model()

    elif name in ['vnet']:
        model = model()

    elif name in ['iunet']:
        model = model(
              32, # input channels or input shape, must be at least as large as slice_fraction
              dim=3, # 3D data input
              architecture=[3]*5, # 7*10*2=140 convolutional layers
              create_module_fn=create_double_module,
              slice_fraction = 4, # Fraction of 
              learnable_downsampling=True, # Otherwise, 3D Haar wavelets are used
              disable_custom_gradient=False)

    elif name in ['unet', 'unet_ct_dsv']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      feature_scale=feature_scale,
                      is_deconv=False)
    elif name in ['unet_nonlocal']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      is_deconv=False,
                      nonlocal_mode=nonlocal_mode,
                      feature_scale=feature_scale)
    elif name in ['unet_grid_gating',
                  'unet_ct_single_att_dsv',
                  'unet_ct_multi_att_dsv']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      nonlocal_mode=nonlocal_mode,
                      feature_scale=feature_scale,
                      attention_dsample=attention_dsample,
                      is_deconv=False)
    elif name in ['sononet','sononet2']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      feature_scale=feature_scale)
    elif name in ['sononet_grid_attention']:
        model = model(n_classes=n_classes,
                      is_batchnorm=True,
                      in_channels=in_channels,
                      feature_scale=feature_scale,
                      nonlocal_mode=nonlocal_mode,
                      aggregation_mode=aggregation_mode)
    else:
        raise 'Model {} not available'.format(name)

    return model


def _get_model_instance(name, tensor_dim):
    return {
        'iunet' : {'3D':iUNet},
        'fully_reversible':{'3D':FullyReversible},
        'vnet':{'3D':VNet},
        'revunet_dsv':{'3D':NoNewReversible_dsv},
        'revunet_wide_dsv':{'3D':NoNewReversible_wide_dsv},
        'revunet_deep_dsv':{'3D':NoNewReversible_deep_dsv},
        'revunet':{'3D':NoNewReversible},
        'revunet_big':{'3D':NoNewReversible_big},
        'unet':{'2D': unet_2D, '3D': unet_3D},
        'unet_nonlocal':{'2D': unet_nonlocal_2D, '3D': unet_nonlocal_3D},
        'unet_grid_gating': {'3D': unet_grid_attention_3D},
        'unet_ct_dsv': {'3D': unet_CT_dsv_3D},
        'unet_ct_single_att_dsv': {'3D': unet_CT_single_att_dsv_3D},
        'unet_ct_multi_att_dsv': {'3D': unet_CT_multi_att_dsv_3D},
        'sononet': {'2D': sononet},
        'sononet2': {'2D': sononet2},
        'sononet_grid_attention': {'2D': sononet_grid_attention}
    }[name][tensor_dim]
