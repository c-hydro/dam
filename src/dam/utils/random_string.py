from random import randint
from datetime import datetime

# -------------------------------------------------------------------------------------
# Method to create a random string
def random_string(string_root='temporary', string_separetor='_', rand_min=0, rand_max=1000):

    # Rand number
    rand_n = str(randint(rand_min, rand_max))
    # Rand time
    rand_time = datetime.now().strftime('%Y%m%d-%H%M%S_%f')
    # Rand string
    rand_string = string_separetor.join([string_root, rand_time, rand_n])

    return rand_string
# -------------------------------------------------------------------------------------