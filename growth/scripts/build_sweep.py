from growth.sweep.arguments import SweepArguments
from growth.sweep.sweep import Sweep


# ======================== PARSE SCRIPT ARGUMENTS =============================

args = SweepArguments(description='Parameter sweep arguments.')
density = args['density']
batch_size = args['batch_size']
division_rate = args['division_rate']
population = args['population']

# ============================= RUN SCRIPT ====================================

# instantiate sweep object
sweep = Sweep(density=density,
              batch_size=batch_size,
              division_rate=division_rate,
              population=population)

# build sweep
sweep.build(
    directory=args['path'],
    batch_size=args['batch_size'],
    save_history=args['save_history'],
    walltime=args['walltime'],
    allocation=args['allocation'])

"""
LOCAL TEST:

python build_sweep.py ../../simulations/ -b 3 -d 1 -s 1

"""
