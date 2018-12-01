from growth.sweep.arguments import SweepArguments
from growth.sweep.sweep import Sweep


# ======================== PARSE SCRIPT ARGUMENTS =============================

args = SweepArguments(description='Arguments for a growth parameter sweep.')

first_start = args['first_start']
last_start = args['last_start']
duration = args['duration']
num_periods = args['num_periods']
division_rate = args['division_rate']
min_rate = args['min_rate']
max_rate = args['max_rate']
num_rates = args['num_rates']
min_population = args['min_population']
num_replicates = args['num_replicates']

# ============================= RUN SCRIPT ====================================

# instantiate sweep object
sweep = Sweep(

    # argument defining growth rate
    division_rate=division_rate,

    # arguments defining recombination period
    duration=duration,
    first_start=first_start,
    last_start=last_start,
    num_periods=num_periods,

    # arguments defining recombination rate
    min_rate=min_rate,
    max_rate=max_rate,
    num_rates=num_rates,

    # arguments defining simulation size
    min_population=min_population,
    num_replicates=num_replicates)

# build sweep
sweep.build(
    directory=args['path'],
    save_history=args['save_history'],
    walltime=args['walltime'],
    cores=args['cores'],
    memory=args['memory'],
    allocation=args['allocation'])
