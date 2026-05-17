

# all epsilon functions are a function of updates done, not episodes
# that way epsilon doesn't start decaying while the buffer fills up

def linear_decay(updates: int, start_eps: float, max_eps: float, episodes_to_max: int):

    return min(max_eps, (updates/episodes_to_max) * (max_eps - start_eps) + start_eps)


funcs = {
    'linear_decay': linear_decay
}

def get_epsilon_function(name: str, *args, **kwargs):

    return lambda x: funcs[name](x, *args, **kwargs)

