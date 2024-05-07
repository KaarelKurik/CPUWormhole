# CPUWormhole

This is a (terribly slow) CPU-based raytracer for drawing
a wormhole. Intended to be a sanity-check before I start
implementing this logic on a GPU, which is a more unfamiliar
setting to me.

The wormholes drawn here are not based on GR, instead an arbitrarily
chosen metric interpolation of annular patches of flat space. (More precisely,
we interpolate the inverse metrics, because we need the inverse metrics at every step of the calculation, but the metrics only at the start and end,
so interpolating the inverse metrics directly is less expensive.)
At time of writing,
they exhibit some exotic properties which may or may not be the result
of implementation bugs, like deflecting skimming rays outward rather than
bending them around the hole.

The core raytracing loop is based on https://arxiv.org/abs/1609.02212, which
gives us a symplectic integrator even though our Hamiltonian does not appear
to split. This has previously been applied to the problem of geodesic integration in https://arxiv.org/pdf/2010.02237.
The Hamiltonian in this setting is $`H(q,p) = \frac{1}{2} g^{ij}(q) p_i p_j`$.

No symmetries were ~~harmed~~ used in the making of these images. Spherical
wormholes can be rendered very efficiently by making use of spherical 
symmetry, but our logic is supposed to be general over wormholes of different
shapes, like toruses, tetrahedra, or monkeys, and these do not generically have symmetries we could take advantage of.