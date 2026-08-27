

# all epsilon functions are a function of updates done, not episodes
# that way epsilon doesn't start decaying while the buffer fills up

def linear_decay(updates: int, start: float, max: float, episodes_to_max: int):

    return min(max, (updates/episodes_to_max) * (max - start) + start)


funcs = {
    'linear_decay': linear_decay
}

def get_epsilon_function(name: str, *args, **kwargs):

    return lambda x: funcs[name](x, *args, **kwargs)

