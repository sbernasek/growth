# Tissue Growth Simulation

This python package may be used to simulate the 2-D growth of a synthetic cell culture subject to mitotic recombination. Please note that

**<font color='red'>Please note that this package is not intended for wide-spread distribution. We are only making the code available, along with minimal usage instructions, so that other researchers may reproduce the results published in our Fly-QMA manuscript. Code documentation is therefore incomplete and in many cases missing entirely, and we anticipate that future support and development will be limited. That said, we are more than happy to address any specific questions or concerns that might arise, so don't hesitate to reach out via [GitHub](https://github.com/sebastianbernasek).</font>**


Dependencies
============

 - Python 3.6+

Required:

 - [Scipy](https://www.scipy.org/)
 - [Pandas](https://pandas.pydata.org/)
 - [NetworkX](https://networkx.github.io/)
 - [Matplotlib](https://matplotlib.org/)


Installation
============

The simplest method is to download the latest [distribution](https://github.com/sebastianbernasek/growth/archive/v0.1.tar.gz) and install via ``pip``:

    pip install growth-0.1.tar.gz


Example Usage
=============

To run a basic tissue growth simulation:

    from growth import Culture

    # define the simulation size
    num_cells = 100

    # initialize a synthetic cell culture
    culture = Culture(reference_population=num_cells)

    # run the simulation
    culture.grow(min_population=num_cells, division_rate=0.1, recombination_rate=0.1)

To generate synthetic fluorescence measurements:

    measurements = culture.measure(ambiguity=0.1)


We have also provided a barebones [tutorial](https://github.com/sebastianbernasek/growth/blob/master/tutorial.ipynb) that walks through the steps needed to perform a single growth simulation, visualize the resultant synthetic cell culture, and generate synthetic fluorescence measurement data.


Authors
=======

[Amaral Lab](https://amaral.northwestern.edu/)
