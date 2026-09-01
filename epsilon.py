from torch import e

# all epsilon functions are a function of updates done, not episodes
# that way epsilon doesn't start decaying while the buffer fills up


def linear_decay(updates: int, start: float, max: float, updates_to_max: int):

    return min(max, (updates/updates_to_max) * (max - start) + start)


def sigmoid_wall(updates: int, scale: float, weight: float, offset: float):

    # scale is the expected updates over which to broadcast the shape
    # weight determines the sharpness of the shape
    # offset determines the location of the sigmoid center
    
    sigmoid_input = weight * updates / scale - offset

    return 1 / (1 + e**(-sigmoid_input))


def constant(updates: int, value: float):
    return value


funcs = {
    'linear_decay': linear_decay,
    'sigmoid_wall': sigmoid_wall,
    'constant': constant
}

def get_epsilon_function(name: str, *args, **kwargs):

    return lambda x: funcs[name](x, *args, **kwargs)

