This code aims to formulate and solve the NLSE in a Bragg grating. Require CUDA.jl

This is different from the standard NLSE solving because a z-dependant operator need to be introduced (the grating dispersion operator).
The code is made to run on GPU (very fast!).

In the .jl file you have 2 key function, gpu_NLSE_BG that does the formulation (grating operator) and solving (split step) that gives the output spectrum corresponding to an input spectrum.
Side products also given by this function: the γ parameter, the dispersion operator and the temporal output.

To build the input spectrum, the second key function "input_pulse" has been made.

An example is available at the end of the file (commented) if you want to use the code with realistic parameters directly.
