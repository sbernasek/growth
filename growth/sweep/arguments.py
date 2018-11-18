from os import getcwd
from argparse import ArgumentParser, ArgumentTypeError


# ======================== ARGUMENT TYPE CASTING ==============================

def str2bool(arg):
     """ Convert <arg> to boolean. """
     if arg.lower() in ('yes', 'true', 't', 'y', '1'):
          return True
     elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
          return False
     else:
          raise ArgumentTypeError('Boolean value expected.')


# ======================== ARGUMENT PARSING ===================================


class RunArguments(ArgumentParser):
     """ Argument handler for run scripts. """

     def __init__(self, **kwargs):
          super().__init__(**kwargs)
          self.add_arguments()
          self.parse()

     def __getitem__(self, key):
          """ Returns <key> argument value. """
          return self.args[key]

     def add_arguments(self):
          """ Add arguments. """

          # add position argument for path
          self.add_argument(
               'path',
               nargs='?',
               default=getcwd())

          # add keyword argument for saving history
          self.add_argument(
               '-s', '--save_history',
               help='Save growth history.',
               type=str2bool,
               default=True,
               required=False)

     def parse(self):
          """ Parse arguments. """
          self.args = vars(self.parse_args())


class SweepArguments(RunArguments):
     """ Argument handler for parameter sweeps. """

     def add_arguments(self):
          """ Add arguments. """

          super().add_arguments()

          # add keyword argument for sweep density
          self.add_argument('-d', '--density',
                              help='Parameter range density.',
                              type=int,
                              default=11,
                              required=False)

          # add keyword argument for sweep density
          self.add_argument('-b', '--batch_size',
                              help='Replicates per parameter set.',
                              type=int,
                              default=10,
                              required=False)

          # add keyword argument for project allocation
          self.add_argument('-w', '--walltime',
                              help='Estimated run time.',
                              type=int,
                              default=10,
                              required=False)

          # add keyword argument for project allocation
          self.add_argument('-A', '--allocation',
                              help='Project allocation.',
                              type=str,
                              default='p30653',
                              required=False)
