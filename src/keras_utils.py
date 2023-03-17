"""Interface converters for Keras 1 support in Keras 2.
"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import six
import warnings
import functools
import numpy as np

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils.generic_utils import to_list

def generate_legacy_interface(allowed_positional_args=None,
                              conversions=None,
                              preprocessor=None,
                              value_conversions=None,
                              object_type='class'):
    if allowed_positional_args is None:
        check_positional_args = False
    else:
        check_positional_args = True
    allowed_positional_args = allowed_positional_args or []
    conversions = conversions or []
    value_conversions = value_conversions or []

    def legacy_support(func):
        @six.wraps(func)
        def wrapper(*args, **kwargs):
            if object_type == 'class':
                object_name = args[0].__class__.__name__
            else:
                object_name = func.__name__
            if preprocessor:
                args, kwargs, converted = preprocessor(args, kwargs)
            else:
                converted = []
            if check_positional_args:
                if len(args) > len(allowed_positional_args) + 1:
                    raise TypeError('`' + object_name +
                                    '` can accept only ' +
                                    str(len(allowed_positional_args)) +
                                    ' positional arguments ' +
                                    str(tuple(allowed_positional_args)) +
                                    ', but you passed the following '
                                    'positional arguments: ' +
                                    str(list(args[1:])))
            for key in value_conversions:
                if key in kwargs:
                    old_value = kwargs[key]
                    if old_value in value_conversions[key]:
                        kwargs[key] = value_conversions[key][old_value]
            for old_name, new_name in conversions:
                if old_name in kwargs:
                    value = kwargs.pop(old_name)
                    if new_name in kwargs:
                        raise_duplicate_arg_error(old_name, new_name)
                    kwargs[new_name] = value
                    converted.append((new_name, old_name))
            if converted:
                signature = '`' + object_name + '('
                for i, value in enumerate(args[1:]):
                    if isinstance(value, six.string_types):
                        signature += '"' + value + '"'
                    else:
                        if isinstance(value, np.ndarray):
                            str_val = 'array'
                        else:
                            str_val = str(value)
                        if len(str_val) > 10:
                            str_val = str_val[:10] + '...'
                        signature += str_val
                    if i < len(args[1:]) - 1 or kwargs:
                        signature += ', '
                for i, (name, value) in enumerate(kwargs.items()):
                    signature += name + '='
                    if isinstance(value, six.string_types):
                        signature += '"' + value + '"'
                    else:
                        if isinstance(value, np.ndarray):
                            str_val = 'array'
                        else:
                            str_val = str(value)
                        if len(str_val) > 10:
                            str_val = str_val[:10] + '...'
                        signature += str_val
                    if i < len(kwargs) - 1:
                        signature += ', '
                signature += ')`'
                warnings.warn('Update your `' + object_name + '` call to the ' +
                              'Keras 2 API: ' + signature, stacklevel=2)
            return func(*args, **kwargs)
        wrapper._original_function = func
        return wrapper
    return legacy_support


generate_legacy_method_interface = functools.partial(generate_legacy_interface,
                                                     object_type='method')


def raise_duplicate_arg_error(old_arg, new_arg):
    raise TypeError('For the `' + new_arg + '` argument, '
                    'the layer received both '
                    'the legacy keyword argument '
                    '`' + old_arg + '` and the Keras 2 keyword argument '
                    '`' + new_arg + '`. Stick to the latter!')


legacy_dense_support = generate_legacy_interface(
    allowed_positional_args=['units'],
    conversions=[('output_dim', 'units'),
                 ('init', 'kernel_initializer'),
                 ('W_regularizer', 'kernel_regularizer'),
                 ('b_regularizer', 'bias_regularizer'),
                 ('W_constraint', 'kernel_constraint'),
                 ('b_constraint', 'bias_constraint'),
                 ('bias', 'use_bias')])

legacy_dropout_support = generate_legacy_interface(
    allowed_positional_args=['rate', 'noise_shape', 'seed'],
    conversions=[('p', 'rate')])


def embedding_kwargs_preprocessor(args, kwargs):
    converted = []
    if 'dropout' in kwargs:
        kwargs.pop('dropout')
        warnings.warn('The `dropout` argument is no longer support in `Embedding`. '
                      'You can apply a `keras.layers.SpatialDropout1D` layer '
                      'right after the `Embedding` layer to get the same behavior.',
                      stacklevel=3)
    return args, kwargs, converted

legacy_embedding_support = generate_legacy_interface(
    allowed_positional_args=['input_dim', 'output_dim'],
    conversions=[('init', 'embeddings_initializer'),
                 ('W_regularizer', 'embeddings_regularizer'),
                 ('W_constraint', 'embeddings_constraint')],
    preprocessor=embedding_kwargs_preprocessor)

legacy_pooling1d_support = generate_legacy_interface(
    allowed_positional_args=['pool_size', 'strides', 'padding'],
    conversions=[('pool_length', 'pool_size'),
                 ('stride', 'strides'),
                 ('border_mode', 'padding')])

legacy_prelu_support = generate_legacy_interface(
    allowed_positional_args=['alpha_initializer'],
    conversions=[('init', 'alpha_initializer')])


legacy_gaussiannoise_support = generate_legacy_interface(
    allowed_positional_args=['stddev'],
    conversions=[('sigma', 'stddev')])


def recurrent_args_preprocessor(args, kwargs):
    converted = []
    if 'forget_bias_init' in kwargs:
        if kwargs['forget_bias_init'] == 'one':
            kwargs.pop('forget_bias_init')
            kwargs['unit_forget_bias'] = True
            converted.append(('forget_bias_init', 'unit_forget_bias'))
        else:
            kwargs.pop('forget_bias_init')
            warnings.warn('The `forget_bias_init` argument '
                          'has been ignored. Use `unit_forget_bias=True` '
                          'instead to initialize with ones.', stacklevel=3)
    if 'input_dim' in kwargs:
        input_length = kwargs.pop('input_length', None)
        input_dim = kwargs.pop('input_dim')
        input_shape = (input_length, input_dim)
        kwargs['input_shape'] = input_shape
        converted.append(('input_dim', 'input_shape'))
        warnings.warn('The `input_dim` and `input_length` arguments '
                      'in recurrent layers are deprecated. '
                      'Use `input_shape` instead.', stacklevel=3)
    return args, kwargs, converted

legacy_recurrent_support = generate_legacy_interface(
    allowed_positional_args=['units'],
    conversions=[('output_dim', 'units'),
                 ('init', 'kernel_initializer'),
                 ('inner_init', 'recurrent_initializer'),
                 ('inner_activation', 'recurrent_activation'),
                 ('W_regularizer', 'kernel_regularizer'),
                 ('b_regularizer', 'bias_regularizer'),
                 ('U_regularizer', 'recurrent_regularizer'),
                 ('dropout_W', 'dropout'),
                 ('dropout_U', 'recurrent_dropout'),
                 ('consume_less', 'implementation')],
    value_conversions={'consume_less': {'cpu': 0,
                                        'mem': 1,
                                        'gpu': 2}},
    preprocessor=recurrent_args_preprocessor)

legacy_gaussiandropout_support = generate_legacy_interface(
    allowed_positional_args=['rate'],
    conversions=[('p', 'rate')])

legacy_pooling2d_support = generate_legacy_interface(
    allowed_positional_args=['pool_size', 'strides', 'padding'],
    conversions=[('border_mode', 'padding'),
                 ('dim_ordering', 'data_format')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}})

legacy_pooling3d_support = generate_legacy_interface(
    allowed_positional_args=['pool_size', 'strides', 'padding'],
    conversions=[('border_mode', 'padding'),
                 ('dim_ordering', 'data_format')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}})

legacy_global_pooling_support = generate_legacy_interface(
    conversions=[('dim_ordering', 'data_format')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}})

legacy_upsampling1d_support = generate_legacy_interface(
    allowed_positional_args=['size'],
    conversions=[('length', 'size')])

legacy_upsampling2d_support = generate_legacy_interface(
    allowed_positional_args=['size'],
    conversions=[('dim_ordering', 'data_format')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}})

legacy_upsampling3d_support = generate_legacy_interface(
    allowed_positional_args=['size'],
    conversions=[('dim_ordering', 'data_format')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}})


def conv1d_args_preprocessor(args, kwargs):
    converted = []
    if 'input_dim' in kwargs:
        if 'input_length' in kwargs:
            length = kwargs.pop('input_length')
        else:
            length = None
        input_shape = (length, kwargs.pop('input_dim'))
        kwargs['input_shape'] = input_shape
        converted.append(('input_shape', 'input_dim'))
    return args, kwargs, converted

legacy_conv1d_support = generate_legacy_interface(
    allowed_positional_args=['filters', 'kernel_size'],
    conversions=[('nb_filter', 'filters'),
                 ('filter_length', 'kernel_size'),
                 ('subsample_length', 'strides'),
                 ('border_mode', 'padding'),
                 ('init', 'kernel_initializer'),
                 ('W_regularizer', 'kernel_regularizer'),
                 ('b_regularizer', 'bias_regularizer'),
                 ('W_constraint', 'kernel_constraint'),
                 ('b_constraint', 'bias_constraint'),
                 ('bias', 'use_bias')],
    preprocessor=conv1d_args_preprocessor)


def conv2d_args_preprocessor(args, kwargs):
    converted = []
    if len(args) > 4:
        raise TypeError('Layer can receive at most 3 positional arguments.')
    elif len(args) == 4:
        if isinstance(args[2], int) and isinstance(args[3], int):
            new_keywords = ['padding', 'strides', 'data_format']
            for kwd in new_keywords:
                if kwd in kwargs:
                    raise ValueError(
                        'It seems that you are using the Keras 2 '
                        'and you are passing both `kernel_size` and `strides` '
                        'as integer positional arguments. For safety reasons, '
                        'this is disallowed. Pass `strides` '
                        'as a keyword argument instead.')
            kernel_size = (args[2], args[3])
            args = [args[0], args[1], kernel_size]
            converted.append(('kernel_size', 'nb_row/nb_col'))
    elif len(args) == 3 and isinstance(args[2], int):
        if 'nb_col' in kwargs:
            kernel_size = (args[2], kwargs.pop('nb_col'))
            args = [args[0], args[1], kernel_size]
            converted.append(('kernel_size', 'nb_row/nb_col'))
    elif len(args) == 2:
        if 'nb_row' in kwargs and 'nb_col' in kwargs:
            kernel_size = (kwargs.pop('nb_row'), kwargs.pop('nb_col'))
            args = [args[0], args[1], kernel_size]
            converted.append(('kernel_size', 'nb_row/nb_col'))
    elif len(args) == 1:
        if 'nb_row' in kwargs and 'nb_col' in kwargs:
            kernel_size = (kwargs.pop('nb_row'), kwargs.pop('nb_col'))
            kwargs['kernel_size'] = kernel_size
            converted.append(('kernel_size', 'nb_row/nb_col'))
    return args, kwargs, converted

legacy_conv2d_support = generate_legacy_interface(
    allowed_positional_args=['filters', 'kernel_size'],
    conversions=[('nb_filter', 'filters'),
                 ('subsample', 'strides'),
                 ('border_mode', 'padding'),
                 ('dim_ordering', 'data_format'),
                 ('init', 'kernel_initializer'),
                 ('W_regularizer', 'kernel_regularizer'),
                 ('b_regularizer', 'bias_regularizer'),
                 ('W_constraint', 'kernel_constraint'),
                 ('b_constraint', 'bias_constraint'),
                 ('bias', 'use_bias')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}},
    preprocessor=conv2d_args_preprocessor)


def separable_conv2d_args_preprocessor(args, kwargs):
    converted = []
    if 'init' in kwargs:
        init = kwargs.pop('init')
        kwargs['depthwise_initializer'] = init
        kwargs['pointwise_initializer'] = init
        converted.append(('init', 'depthwise_initializer/pointwise_initializer'))
    args, kwargs, _converted = conv2d_args_preprocessor(args, kwargs)
    return args, kwargs, converted + _converted

legacy_separable_conv2d_support = generate_legacy_interface(
    allowed_positional_args=['filters', 'kernel_size'],
    conversions=[('nb_filter', 'filters'),
                 ('subsample', 'strides'),
                 ('border_mode', 'padding'),
                 ('dim_ordering', 'data_format'),
                 ('b_regularizer', 'bias_regularizer'),
                 ('b_constraint', 'bias_constraint'),
                 ('bias', 'use_bias')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}},
    preprocessor=separable_conv2d_args_preprocessor)


def deconv2d_args_preprocessor(args, kwargs):
    converted = []
    if len(args) == 5:
        if isinstance(args[4], tuple):
            args = args[:-1]
            converted.append(('output_shape', None))
    if 'output_shape' in kwargs:
        kwargs.pop('output_shape')
        converted.append(('output_shape', None))
    args, kwargs, _converted = conv2d_args_preprocessor(args, kwargs)
    return args, kwargs, converted + _converted

legacy_deconv2d_support = generate_legacy_interface(
    allowed_positional_args=['filters', 'kernel_size'],
    conversions=[('nb_filter', 'filters'),
                 ('subsample', 'strides'),
                 ('border_mode', 'padding'),
                 ('dim_ordering', 'data_format'),
                 ('init', 'kernel_initializer'),
                 ('W_regularizer', 'kernel_regularizer'),
                 ('b_regularizer', 'bias_regularizer'),
                 ('W_constraint', 'kernel_constraint'),
                 ('b_constraint', 'bias_constraint'),
                 ('bias', 'use_bias')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}},
    preprocessor=deconv2d_args_preprocessor)


def conv3d_args_preprocessor(args, kwargs):
    converted = []
    if len(args) > 5:
        raise TypeError('Layer can receive at most 4 positional arguments.')
    if len(args) == 5:
        if all([isinstance(x, int) for x in args[2:5]]):
            kernel_size = (args[2], args[3], args[4])
            args = [args[0], args[1], kernel_size]
            converted.append(('kernel_size', 'kernel_dim*'))
    elif len(args) == 4 and isinstance(args[3], int):
        if isinstance(args[2], int) and isinstance(args[3], int):
            new_keywords = ['padding', 'strides', 'data_format']
            for kwd in new_keywords:
                if kwd in kwargs:
                    raise ValueError(
                        'It seems that you are using the Keras 2 '
                        'and you are passing both `kernel_size` and `strides` '
                        'as integer positional arguments. For safety reasons, '
                        'this is disallowed. Pass `strides` '
                        'as a keyword argument instead.')
        if 'kernel_dim3' in kwargs:
            kernel_size = (args[2], args[3], kwargs.pop('kernel_dim3'))
            args = [args[0], args[1], kernel_size]
            converted.append(('kernel_size', 'kernel_dim*'))
    elif len(args) == 3:
        if all([x in kwargs for x in ['kernel_dim2', 'kernel_dim3']]):
            kernel_size = (args[2],
                           kwargs.pop('kernel_dim2'),
                           kwargs.pop('kernel_dim3'))
            args = [args[0], args[1], kernel_size]
            converted.append(('kernel_size', 'kernel_dim*'))
    elif len(args) == 2:
        if all([x in kwargs for x in ['kernel_dim1', 'kernel_dim2', 'kernel_dim3']]):
            kernel_size = (kwargs.pop('kernel_dim1'),
                           kwargs.pop('kernel_dim2'),
                           kwargs.pop('kernel_dim3'))
            args = [args[0], args[1], kernel_size]
            converted.append(('kernel_size', 'kernel_dim*'))
    elif len(args) == 1:
        if all([x in kwargs for x in ['kernel_dim1', 'kernel_dim2', 'kernel_dim3']]):
            kernel_size = (kwargs.pop('kernel_dim1'),
                           kwargs.pop('kernel_dim2'),
                           kwargs.pop('kernel_dim3'))
            kwargs['kernel_size'] = kernel_size
            converted.append(('kernel_size', 'nb_row/nb_col'))
    return args, kwargs, converted

legacy_conv3d_support = generate_legacy_interface(
    allowed_positional_args=['filters', 'kernel_size'],
    conversions=[('nb_filter', 'filters'),
                 ('subsample', 'strides'),
                 ('border_mode', 'padding'),
                 ('dim_ordering', 'data_format'),
                 ('init', 'kernel_initializer'),
                 ('W_regularizer', 'kernel_regularizer'),
                 ('b_regularizer', 'bias_regularizer'),
                 ('W_constraint', 'kernel_constraint'),
                 ('b_constraint', 'bias_constraint'),
                 ('bias', 'use_bias')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}},
    preprocessor=conv3d_args_preprocessor)


def batchnorm_args_preprocessor(args, kwargs):
    converted = []
    if len(args) > 1:
        raise TypeError('The `BatchNormalization` layer '
                        'does not accept positional arguments. '
                        'Use keyword arguments instead.')
    if 'mode' in kwargs:
        value = kwargs.pop('mode')
        if value != 0:
            raise TypeError('The `mode` argument of `BatchNormalization` '
                            'no longer exists. `mode=1` and `mode=2` '
                            'are no longer supported.')
        converted.append(('mode', None))
    return args, kwargs, converted


def convlstm2d_args_preprocessor(args, kwargs):
    converted = []
    if 'forget_bias_init' in kwargs:
        value = kwargs.pop('forget_bias_init')
        if value == 'one':
            kwargs['unit_forget_bias'] = True
            converted.append(('forget_bias_init', 'unit_forget_bias'))
        else:
            warnings.warn('The `forget_bias_init` argument '
                          'has been ignored. Use `unit_forget_bias=True` '
                          'instead to initialize with ones.', stacklevel=3)
    args, kwargs, _converted = conv2d_args_preprocessor(args, kwargs)
    return args, kwargs, converted + _converted

legacy_convlstm2d_support = generate_legacy_interface(
    allowed_positional_args=['filters', 'kernel_size'],
    conversions=[('nb_filter', 'filters'),
                 ('subsample', 'strides'),
                 ('border_mode', 'padding'),
                 ('dim_ordering', 'data_format'),
                 ('init', 'kernel_initializer'),
                 ('inner_init', 'recurrent_initializer'),
                 ('W_regularizer', 'kernel_regularizer'),
                 ('U_regularizer', 'recurrent_regularizer'),
                 ('b_regularizer', 'bias_regularizer'),
                 ('inner_activation', 'recurrent_activation'),
                 ('dropout_W', 'dropout'),
                 ('dropout_U', 'recurrent_dropout'),
                 ('bias', 'use_bias')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}},
    preprocessor=convlstm2d_args_preprocessor)

legacy_batchnorm_support = generate_legacy_interface(
    allowed_positional_args=[],
    conversions=[('beta_init', 'beta_initializer'),
                 ('gamma_init', 'gamma_initializer')],
    preprocessor=batchnorm_args_preprocessor)


def zeropadding2d_args_preprocessor(args, kwargs):
    converted = []
    if 'padding' in kwargs and isinstance(kwargs['padding'], dict):
        if set(kwargs['padding'].keys()) <= {'top_pad', 'bottom_pad',
                                             'left_pad', 'right_pad'}:
            top_pad = kwargs['padding'].get('top_pad', 0)
            bottom_pad = kwargs['padding'].get('bottom_pad', 0)
            left_pad = kwargs['padding'].get('left_pad', 0)
            right_pad = kwargs['padding'].get('right_pad', 0)
            kwargs['padding'] = ((top_pad, bottom_pad), (left_pad, right_pad))
            warnings.warn('The `padding` argument in the Keras 2 API no longer'
                          'accepts dict types. You can now input argument as: '
                          '`padding=(top_pad, bottom_pad, left_pad, right_pad)`.',
                          stacklevel=3)
    elif len(args) == 2 and isinstance(args[1], dict):
        if set(args[1].keys()) <= {'top_pad', 'bottom_pad',
                                   'left_pad', 'right_pad'}:
            top_pad = args[1].get('top_pad', 0)
            bottom_pad = args[1].get('bottom_pad', 0)
            left_pad = args[1].get('left_pad', 0)
            right_pad = args[1].get('right_pad', 0)
            args = (args[0], ((top_pad, bottom_pad), (left_pad, right_pad)))
            warnings.warn('The `padding` argument in the Keras 2 API no longer'
                          'accepts dict types. You can now input argument as: '
                          '`padding=((top_pad, bottom_pad), (left_pad, right_pad))`',
                          stacklevel=3)
    return args, kwargs, converted

legacy_zeropadding2d_support = generate_legacy_interface(
    allowed_positional_args=['padding'],
    conversions=[('dim_ordering', 'data_format')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}},
    preprocessor=zeropadding2d_args_preprocessor)

legacy_zeropadding3d_support = generate_legacy_interface(
    allowed_positional_args=['padding'],
    conversions=[('dim_ordering', 'data_format')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}})

legacy_cropping2d_support = generate_legacy_interface(
    allowed_positional_args=['cropping'],
    conversions=[('dim_ordering', 'data_format')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}})

legacy_cropping3d_support = generate_legacy_interface(
    allowed_positional_args=['cropping'],
    conversions=[('dim_ordering', 'data_format')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}})

legacy_spatialdropout1d_support = generate_legacy_interface(
    allowed_positional_args=['rate'],
    conversions=[('p', 'rate')])

legacy_spatialdropoutNd_support = generate_legacy_interface(
    allowed_positional_args=['rate'],
    conversions=[('p', 'rate'),
                 ('dim_ordering', 'data_format')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}})

legacy_lambda_support = generate_legacy_interface(
    allowed_positional_args=['function', 'output_shape'])


# Model methods

def generator_methods_args_preprocessor(args, kwargs):
    converted = []
    if len(args) < 3:
        if 'samples_per_epoch' in kwargs:
            samples_per_epoch = kwargs.pop('samples_per_epoch')
            if len(args) > 1:
                generator = args[1]
            else:
                generator = kwargs['generator']
            if hasattr(generator, 'batch_size'):
                kwargs['steps_per_epoch'] = samples_per_epoch // generator.batch_size
            else:
                kwargs['steps_per_epoch'] = samples_per_epoch
            converted.append(('samples_per_epoch', 'steps_per_epoch'))

    keras1_args = {'samples_per_epoch', 'val_samples',
                   'nb_epoch', 'nb_val_samples', 'nb_worker'}
    if keras1_args.intersection(kwargs.keys()):
        warnings.warn('The semantics of the Keras 2 argument '
                      '`steps_per_epoch` is not the same as the '
                      'Keras 1 argument `samples_per_epoch`. '
                      '`steps_per_epoch` is the number of batches '
                      'to draw from the generator at each epoch. '
                      'Basically steps_per_epoch = samples_per_epoch/batch_size. '
                      'Similarly `nb_val_samples`->`validation_steps` and '
                      '`val_samples`->`steps` arguments have changed. '
                      'Update your method calls accordingly.', stacklevel=3)

    return args, kwargs, converted


legacy_generator_methods_support = generate_legacy_method_interface(
    allowed_positional_args=['generator', 'steps_per_epoch', 'epochs'],
    conversions=[('samples_per_epoch', 'steps_per_epoch'),
                 ('val_samples', 'steps'),
                 ('nb_epoch', 'epochs'),
                 ('nb_val_samples', 'validation_steps'),
                 ('nb_worker', 'workers'),
                 ('pickle_safe', 'use_multiprocessing'),
                 ('max_q_size', 'max_queue_size')],
    preprocessor=generator_methods_args_preprocessor)


legacy_model_constructor_support = generate_legacy_interface(
    allowed_positional_args=None,
    conversions=[('input', 'inputs'),
                 ('output', 'outputs')])

legacy_input_support = generate_legacy_interface(
    allowed_positional_args=None,
    conversions=[('input_dtype', 'dtype')])


def add_weight_args_preprocessing(args, kwargs):
    if len(args) > 1:
        if isinstance(args[1], (tuple, list)):
            kwargs['shape'] = args[1]
            args = (args[0],) + args[2:]
            if len(args) > 1:
                if isinstance(args[1], six.string_types):
                    kwargs['name'] = args[1]
                    args = (args[0],) + args[2:]
    return args, kwargs, []


legacy_add_weight_support = generate_legacy_interface(
    allowed_positional_args=['name', 'shape'],
    preprocessor=add_weight_args_preprocessing)


def get_updates_arg_preprocessing(args, kwargs):
    # Old interface: (params, constraints, loss)
    # New interface: (loss, params)
    if len(args) > 4:
        raise TypeError('`get_update` call received more arguments '
                        'than expected.')
    elif len(args) == 4:
        # Assuming old interface.
        opt, params, _, loss = args
        kwargs['loss'] = loss
        kwargs['params'] = params
        return [opt], kwargs, []
    elif len(args) == 3:
        if isinstance(args[1], (list, tuple)):
            assert isinstance(args[2], dict)
            assert 'loss' in kwargs
            opt, params, _ = args
            kwargs['params'] = params
            return [opt], kwargs, []
    return args, kwargs, []

legacy_get_updates_support = generate_legacy_interface(
    allowed_positional_args=None,
    conversions=[],
    preprocessor=get_updates_arg_preprocessing)



class Recurrent(Layer):
    """Abstract base class for recurrent layers.
    Do not use in a model -- it's not a valid layer!
    Use its children classes `LSTM`, `GRU` and `SimpleRNN` instead.
    All recurrent layers (`LSTM`, `GRU`, `SimpleRNN`) also
    follow the specifications of this class and accept
    the keyword arguments listed below.
    # Example
    ```python
        # as the first layer in a Sequential model
        model = Sequential()
        model.add(LSTM(32, input_shape=(10, 64)))
        # now model.output_shape == (None, 32)
        # note: `None` is the batch dimension.
        # for subsequent layers, no need to specify the input size:
        model.add(LSTM(16))
        # to stack recurrent layers, you must use return_sequences=True
        # on any recurrent layer that feeds into another recurrent layer.
        # note that you only need to specify the input size on the first layer.
        model = Sequential()
        model.add(LSTM(64, input_dim=64, input_length=10, return_sequences=True))
        model.add(LSTM(32, return_sequences=True))
        model.add(LSTM(10))
    ```
    # Arguments
        weights: list of Numpy arrays to set as initial weights.
            The list should have 3 elements, of shapes:
            `[(input_dim, output_dim), (output_dim, output_dim), (output_dim,)]`.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
        implementation: one of {0, 1, or 2}.
            If set to 0, the RNN will use
            an implementation that uses fewer, larger matrix products,
            thus running faster on CPU but consuming more memory.
            If set to 1, the RNN will use more matrix products,
            but smaller ones, thus running slower
            (may actually be faster on GPU) while consuming less memory.
            If set to 2 (LSTM/GRU only),
            the RNN will combine the input gate,
            the forget gate and the output gate into a single matrix,
            enabling more time-efficient parallelization on the GPU.
            Note: RNN dropout must be shared for all gates,
            resulting in a slightly reduced regularization.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        input_length: Length of input sequences, to be specified
            when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).
            Note that if the recurrent layer is not the first layer
            in your model, you would need to specify the input length
            at the level of the first layer
            (e.g. via the `input_shape` argument)
    # Input shapes
        3D tensor with shape `(batch_size, timesteps, input_dim)`,
        (Optional) 2D tensors with shape `(batch_size, output_dim)`.
    # Output shape
        - if `return_state`: a list of tensors. The first tensor is
            the output. The remaining tensors are the last states,
            each with shape `(batch_size, units)`.
        - if `return_sequences`: 3D tensor with shape
            `(batch_size, timesteps, units)`.
        - else, 2D tensor with shape `(batch_size, units)`.
    # Masking
        This layer supports masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
        set to `True`.
    # Note on using statefulness in RNNs
        You can set RNN layers to be 'stateful', which means that the states
        computed for the samples in one batch will be reused as initial states
        for the samples in the next batch. This assumes a one-to-one mapping
        between samples in different successive batches.
        To enable statefulness:
            - specify `stateful=True` in the layer constructor.
            - specify a fixed batch size for your model, by passing
                if sequential model:
                  `batch_input_shape=(...)` to the first layer in your model.
                else for functional model with 1 or more Input layers:
                  `batch_shape=(...)` to all the first layers in your model.
                This is the expected shape of your inputs
                *including the batch size*.
                It should be a tuple of integers, e.g. `(32, 10, 100)`.
            - specify `shuffle=False` when calling fit().
        To reset the states of your model, call `.reset_states()` on either
        a specific layer, or on your entire model.
    # Note on specifying the initial state of RNNs
        You can specify the initial state of RNN layers symbolically by
        calling them with the keyword argument `initial_state`. The value of
        `initial_state` should be a tensor or list of tensors representing
        the initial state of the RNN layer.
        You can specify the initial state of RNN layers numerically by
        calling `reset_states` with the keyword argument `states`. The value of
        `states` should be a numpy array or list of numpy arrays representing
        the initial state of the RNN layer.
    """

    def __init__(self, return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 implementation=0,
                 **kwargs):
        super(Recurrent, self).__init__(**kwargs)
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards

        self.stateful = stateful
        self.unroll = unroll
        self.implementation = implementation
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        self.state_spec = None
        self.dropout = 0
        self.recurrent_dropout = 0

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if self.return_sequences:
            output_shape = (input_shape[0], input_shape[1], self.units)
        else:
            output_shape = (input_shape[0], self.units)

        if self.return_state:
            state_shape = [(input_shape[0], self.units) for _ in self.states]
            return [output_shape] + state_shape
        else:
            return output_shape

    def compute_mask(self, inputs, mask):
        if isinstance(mask, list):
            mask = mask[0]
        output_mask = mask if self.return_sequences else None
        if self.return_state:
            state_mask = [None for _ in self.states]
            return [output_mask] + state_mask
        else:
            return output_mask

    def step(self, inputs, states):
        raise NotImplementedError

    def get_constants(self, inputs, training=None):
        return []

    def get_initial_state(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        # (samples, output_dim)
        initial_state = K.tile(initial_state, [1, self.units])
        initial_state = [initial_state for _ in range(len(self.states))]
        return initial_state

    def preprocess_input(self, inputs, training=None):
        return inputs

    def __call__(self, inputs, initial_state=None, **kwargs):

        # If there are multiple inputs, then
        # they should be the main input and `initial_state`
        # e.g. when loading model from file
        if (isinstance(inputs, (list, tuple))
                and len(inputs) > 1 and initial_state is None):
            initial_state = inputs[1:]
            inputs = inputs[0]

        # If `initial_state` is specified,
        # and if it a Keras tensor,
        # then add it to the inputs and temporarily
        # modify the input spec to include the state.
        if initial_state is None:
            return super(Recurrent, self).__call__(inputs, **kwargs)

        initial_state = to_list(initial_state, allow_tuple=True)

        is_keras_tensor = hasattr(initial_state[0], '_keras_history')
        for tensor in initial_state:
            if hasattr(tensor, '_keras_history') != is_keras_tensor:
                raise ValueError('The initial state of an RNN layer cannot be'
                                 ' specified with a mix of Keras tensors and'
                                 ' non-Keras tensors')

        if is_keras_tensor:
            # Compute the full input spec, including state
            input_spec = self.input_spec
            state_spec = self.state_spec
            input_spec = to_list(input_spec)
            state_spec = to_list(state_spec)
            self.input_spec = input_spec + state_spec

            # Compute the full inputs, including state
            inputs = [inputs] + list(initial_state)

            # Perform the call
            output = super(Recurrent, self).__call__(inputs, **kwargs)

            # Restore original input spec
            self.input_spec = input_spec
            return output
        else:
            kwargs['initial_state'] = initial_state
            return super(Recurrent, self).__call__(inputs, **kwargs)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # input shape: `(samples, time (padded with zeros), input_dim)`
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            initial_state = inputs[1:]
            inputs = inputs[0]
        elif initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)

        if isinstance(mask, list):
            mask = mask[0]

        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')
        input_shape = K.int_shape(inputs)
        timesteps = input_shape[1]
        if self.unroll and timesteps in [None, 1]:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined or equal to 1. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')
        constants = self.get_constants(inputs, training=None)
        preprocessed_input = self.preprocess_input(inputs, training=None)
        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_state,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=timesteps)
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        # Properly set learning phase
        if 0 < self.dropout + self.recurrent_dropout:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        if self.return_state:
            states = to_list(states, allow_tuple=True)
            return [output] + states
        else:
            return output

    def reset_states(self, states=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        batch_size = self.input_spec[0].shape[0]
        if not batch_size:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.')
        # initialize state if None
        if self.states[0] is None:
            self.states = [K.zeros((batch_size, self.units))
                           for _ in self.states]
        elif states is None:
            for state in self.states:
                K.set_value(state, np.zeros((batch_size, self.units)))
        else:
            states = to_list(states, allow_tuple=True)
            if len(states) != len(self.states):
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 str(len(self.states)) + ' states, '
                                 'but it received ' + str(len(states)) +
                                 ' state values. Input received: ' +
                                 str(states))
            for index, (value, state) in enumerate(zip(states, self.states)):
                if value.shape != (batch_size, self.units):
                    raise ValueError('State ' + str(index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected shape=' +
                                     str((batch_size, self.units)) +
                                     ', found shape=' + str(value.shape))
                K.set_value(state, value)

    def get_config(self):
        config = {'return_sequences': self.return_sequences,
                  'return_state': self.return_state,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful,
                  'unroll': self.unroll,
                  'implementation': self.implementation}
        base_config = super(Recurrent, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))