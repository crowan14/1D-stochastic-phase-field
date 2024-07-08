# 1D-stochastic-phase-field

Using principle of minimum expected potential energy to find the damage distribution in a bar with uncertain fracture energy (material parameter). The expectation of the energy is computed using integrals over the spatial and stochastic (parameter) domains. The dependence of the solution on space and the uncertain parameter(s) controlling the material is discretized with a neural network.

The file "one_hidden.m" writes files pertaining to the neural network discretization. These are used in "energy.m" to build an objective function in terms of neural network parameters, which is passed into SQP optimization in the "driver.m" script. A stopping criterion for the optimization is set with "outfun.m" 
