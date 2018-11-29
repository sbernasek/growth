from growth.sweep.arguments import SweepArguments
from growth.sweep.sweep import Sweep


# ======================== PARSE SCRIPT ARGUMENTS =============================

args = SweepArguments(description='Parameter sweep arguments.')
density = args['density']
replicates = args['batch_size']
recombination_duration = args['recombination_duration']
division_rate = args['division_rate']
population = args['population']


# ============================= RUN SCRIPT ====================================

# instantiate sweep object
sweep = Sweep(density=density,
              replicates=replicates,
              recombination_duration=recombination_duration,
              division_rate=division_rate,
              population=population)

# build sweep
sweep.build(
    directory=args['path'],
    save_history=args['save_history'],
    walltime=args['walltime'],
    allocation=args['allocation'])
