from time import time
from growth.sweep.arguments import RunArguments
from growth.sweep.sweep import Sweep


# ======================== PARSE SCRIPT ARGUMENTS =============================

args = RunArguments(description='Aggregation arguments.')

# ============================= RUN SCRIPT ====================================

start = time()
sweep = Sweep.load(args['path'])
sweep.aggregate()
runtime = time() - start

print('AGGREGATION COMPLETED IN {:0.2f} seconds.'.format(runtime))
