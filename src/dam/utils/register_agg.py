global AGG_FUNCTIONS

AGG_FUNCTIONS = {}

def as_agg_function():
    def decorator(func):
        AGG_FUNCTIONS[func.__name__] = func
        return func
    return decorator