import functools

def nested_dict_map(f, x):
    """
    Map f over all leaf of nested dict x
    """

    if not isinstance(x, dict):
        return f(x)
    y = dict()
    for key, value in x.items():
        y[key] = nested_dict_map(f, value)
    return y

def nested_dict_reduce(f, x):
    """
    归约（reduce）的特点：
    将多个值合并成一个值
    是一个迭代的过程
    需要一个二元操作函数（接收两个参数）
    操作必须满足结合律（(a+b)+c = a+(b+c)）
    Map f over all values of nested dict x, and reduce to a single value
    """
    if not isinstance(x, dict):
        return x

    reduced_values = list()
    for value in x.values():
        reduced_values.append(nested_dict_reduce(f, value))
    y = functools.reduce(f, reduced_values)
    return y


def nested_dict_check(f, x):
    bool_dict = nested_dict_map(f, x)
    result = nested_dict_reduce(lambda x, y: x and y, bool_dict)
    return result


"""
# 典型的深度学习模型参数结构
model_params = {
    'encoder': {
        'conv1': {
            'weights': tensor(...),
            'bias': tensor(...)
        },
        'conv2': {
            'weights': tensor(...),
            'bias': tensor(...)
        }
    },
    'decoder': {
        'deconv1': {...},
        'deconv2': {...}
    }
}

# 使用nested_dict_map进行参数操作
# 例如：将所有参数移动到GPU
gpu_params = nested_dict_map(lambda x: x.cuda(), model_params)

# 典型的配置文件结构
config = {
    'training': {
        'batch_size': 32,
        'learning_rate': {
            'initial': 0.001,
            'decay': 0.95
        }
    },
    'model': {
        'architecture': {
            'layers': [64, 128, 256],
            'activation': 'relu'
        }
    }
}

# 使用nested_dict_check验证配置
is_valid = nested_dict_check(
    lambda x: isinstance(x, (int, float, str)), 
    config
)


"""
