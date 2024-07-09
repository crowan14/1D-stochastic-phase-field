# 1D-stochastic-phase-field

Using principle of minimum expected potential energy to find the damage distribution in a bar with uncertain fracture energy (material parameter). The expectation of the energy is computed using integrals over the spatial and stochastic (parameter) domains. The dependence of the solution on space and the uncertain parameter(s) controlling the material is discretized with a neural network. The expectation of the total potential energy is


$$ \langle \Pi \rangle = \int_y \int_x \frac{1}{2}g(\phi) \Big( \frac{\partial u}{\partial x} \Big)^2 - b u + \frac{G(x,y)}{2\ell}\Big( \phi^2 + \Big( \frac{\partial \phi}{\partial x} \Big)^2 \Big) dx dy$$

The file "one_hidden.m" writes files pertaining to the neural network discretization. These are used in "energy.m" to build an objective function in terms of neural network parameters, which is passed into SQP optimization in the "driver.m" script. A stopping criterion for the optimization is set with "outfun.m" 
