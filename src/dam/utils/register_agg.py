import datetime as dt
global AGG_FUNCTIONS

AGG_FUNCTIONS = {}

def as_agg_function(allow_overlap = False):
    def decorator(func):
        def wrapper(input, input_agg, this_agg, **kwargs):
            if not allow_overlap:
                ids_to_keep = remove_overlap(input_agg, ids = True)
                input       = [input[i] for i in ids_to_keep]
                input_agg   = [input_agg[i] for i in ids_to_keep]

            return func(input, input_agg, this_agg, **kwargs)

        wrapper.__name__ = func.__name__
        AGG_FUNCTIONS[func.__name__] = wrapper
        return wrapper
    return decorator

def remove_overlap(input_agg, ids = False):

    input_agg_copy = input_agg.copy()

    input_agg_copy.sort(key = lambda x: x.start)
    nonoverlap_agg = [input_agg_copy.pop()]
    while len(input_agg_copy) > 0:
        this_agg = input_agg_copy.pop()
        if this_agg.end < nonoverlap_agg[-1].start:
            nonoverlap_agg.append(this_agg)

    nonoverlap_agg.sort(key = lambda x: x.start)
    if ids:
        return [input_agg.index(t) for t in nonoverlap_agg]
    else:
        return nonoverlap_agg