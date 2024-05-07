# CPUWormhole

This is a (terribly slow) CPU-based raytracer for drawing
a wormhole. Intended to be a sanity-check before I start
implementing this logic on a GPU, which is a more unfamiliar
setting to me.

The wormholes drawn here are not based on GR, instead an arbitrarily
chosen metric interpolation of annular patches of flat space. At time of writing,
they exhibit some exotic properties which may or may not be the result
of implementation bugs, like deflecting skimming rays outward rather than
bending them around the hole.

The core raytracing loop is based on https://arxiv.org/abs/1609.02212, which
gives us a symplectic integrator even though our Hamiltonian does not appear
to split.
The Hamiltonian in this setting is $`H(q,p) = \frac{1}{2} g^{ij}(q) p_i p_j`$.