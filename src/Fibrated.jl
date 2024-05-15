module CpuWormhole
using LinearAlgebra
using ForwardDiff
using InteractiveUtils
using Images
using Rotations

function project_root_dir()
    dir = @__DIR__
    while !ispath(joinpath(dir, "Project.toml"))
        dir = dirname(dir)
    end
    return dir
end

function cross(x,y)
    @assert length(x) == 3 && length(y) == 3
    [x[2]*y[3]-x[3]*y[2], x[3]*y[1]-x[1]*y[3], x[1]*y[2]-x[2]*y[1]]
end

# [-1,1] in each axis
# z is front
# x is bottom
# y is right
struct SkyBoxTexture
    x::Matrix{RGB{N0f8}}
    nx::Matrix{RGB{N0f8}}
    y::Matrix{RGB{N0f8}}
    ny::Matrix{RGB{N0f8}}
    z::Matrix{RGB{N0f8}}
    nz::Matrix{RGB{N0f8}}
end

struct PQuad{T<:Real}
    corner::Vector{T}
    u::Vector{T}
    v::Vector{T}
    texture::Matrix{RGB{N0f8}}
end

struct SkyBox{T<:Real}
    x::PQuad{T}
    nx::PQuad{T}
    y::PQuad{T}
    ny::PQuad{T}
    z::PQuad{T}
    nz::PQuad{T}
end

function SkyBox(t::SkyBoxTexture)
    # I should probably encode everything as rotations of one face
    z = PQuad{Float64}([-1,-1,1], [2,0,0], [0,2,0], t.z)
    x = PQuad{Float64}([1,-1,1],[0,0,-2],[0,2,0],t.x)
    y = PQuad{Float64}([-1,1,1],[2,0,0],[0,0,-2],t.y)
    nz = PQuad{Float64}([-1,1,-1],[2,0,0],[0,-2,0],t.nz)
    nx = PQuad{Float64}([-1,-1,-1],[0,0,2],[0,2,0],t.nx)
    ny = PQuad{Float64}([-1,-1,-1],[2,0,0],[0,0,2],t.ny)
    SkyBox(x,nx,y,ny,z,nz)
end

abstract type Location{T <: Real} end
abstract type Entry{T <: Real} end

# This ray is in tangent space
# We pass to cotangent space only for pushing
struct Ray{T <: Real}
    q::Vector{T}
    v::Vector{T}
    spot::Location{T}
end

struct Camera{T <: Real}
    centre::Vector{T}
    frame::Matrix{T} # up to positive scalar
    width::Int
    height::Int
    spot::Location{T}
end

function pixel_to_vector(frame::Matrix{T}, w, h, x, y) where {T <: Real}
    normalize(frame * [x-(h+1)/2, y-(w+1)/2, one(T)])
end

function pixel_to_vector(camera::Camera{T}, x, y) where {T <: Real}
    pixel_to_vector(camera.frame, camera.width, camera.height, x, y)
end

function pixel_to_ray(camera::Camera{T}, x, y) where {T <: Real}
    velocity = pixel_to_vector(camera, x, y)
    Ray(camera.centre, velocity, camera.spot)
end

# also checks visibility
# assumes flat space
function vec_quad_coordinates(v::Vector{T}, q::PQuad{T}) where {T <: Real}
    E = [q.u q.v]
    quad_normal = cross(q.u, q.v)
    quad_constant = dot(quad_normal, q.corner)
    λ = quad_constant/dot(quad_normal, v)
    intersec_coords = pinv(E) * (λ * v - q.corner)
    visible = λ > 0 && all(x -> 0 <= x <= 1, intersec_coords)
    intersec_coords, visible
end

# x, y in [0,1]
function get_texel(texture::Matrix{RGB{N0f8}}, coords)
    discretes = [round(Int, a*(as - 1) + 1) for (a,as) in zip(coords, size(texture))]
    texture[discretes...]
end

function get_vec_color(v::Vector{T}, skybox::SkyBox) where {T<:Real}
    for quad in (getfield(skybox, field) for field in fieldnames(SkyBox))
        ic, visible = vec_quad_coordinates(v, quad)
        if visible
            return get_texel(quad.texture, ic)
        end
    end
    nothing
end

function get_ray_color(r::Ray{T}) where {T <: Real}
    get_vec_color(r.v, r.spot.skybox)
end

mutable struct OrientedThroat{T <: Real} <: Location{T}
    ambient_a # Ambient{T}
    ambient_b # Ambient{T}
    entry_a::Entry{T}
    entry_b::Entry{T}
    transition::Function # purely positional transition
    partition::Function # one half of a near-partition of unity. Should be 1 near exterior of entry_a, 0 near interior (i.e. exterior of entry_b)
    opposite::Union{Nothing, OrientedThroat{T}} # should be same throat but viewed from other side, with inverse transition
    ray_pusher::Function
end

struct Ambient{T <: Real} <: Location{T}
    # neighbors::Vector{Ambient{T}}
    throats::Vector{OrientedThroat{T}}
    skybox::SkyBox
    inverse_metric::Function
    ray_pusher::Function
end

struct SphericalEntry{T <: Real} <: Entry{T}
    centre::Vector{T}
    orthoframe::Matrix{T} # should be an orthonormal frame fixing the orientation of the throat
    # specified relative to the implicit frame of ambient space
    outer_throat_radius::T
    inner_throat_radius::T
    ambient::Ambient{T}
end

# rotation in the xy-plane
struct ToroidalEntry{T <: Real} <: Entry{T}
    centre::Vector{T}
    orthoframe::Matrix{T}
    major_radius::T
    outer_minor_radius::T
    inner_minor_radius::T
    ambient::Ambient{T}
end

function sdf(t::ToroidalEntry{T}, p) where {T <: Real}
    ploc = t.orthoframe' * (p - t.centre)  #assumes orthoframe is orthonormal, of course - that's supposed to be the inverse
    partial_norm = norm(@view ploc[1:end-1])
    sqrt(sum(abs2, ploc) + t.major_radius^2 - 2*t.major_radius*partial_norm) - t.outer_minor_radius
end

function smootherstep(x)
    if x <= 0
        zero(x)
    elseif 1 <= x
        one(x)
    else
        6 * x^5 - 15 * x^4 + 10 * x^3
    end
end

# 1 on the outside of a fiber, 0 on the inside
function torus_parameter(t::ToroidalEntry{T}, p) where {T <: Real}
    raw_distance = sdf(t, p) + t.outer_minor_radius
    log(raw_distance/t.inner_minor_radius)/log(t.outer_minor_radius/t.inner_minor_radius)
end

function smooth_torus_parameter(t::ToroidalEntry{T}, p) where {T <: Real}
    smootherstep(torus_parameter(t, p))
end

function smooth_torus_parameter(t::ToroidalEntry{T}) where {T <: Real}
    function out(p)
        smooth_torus_parameter(t,p)
    end
    out
end

function inverse_metric(o::OrientedThroat{T}, x) where {T <: Real}
    y = o.transition(x)
    J = ForwardDiff.jacobian(x->o.opposite.transition(x), y)
    smp = o.partition(x)
    smp * o.ambient_a.inverse_metric(x) + (1-smp) * (J * o.ambient_b.inverse_metric(y) * J')
end

function euclidean_inverse_metric(x::Q) where {T<:Real, Q<:AbstractArray{T}}
    Matrix{T}(I,3,3)
end

function inverse_metric(o::OrientedThroat{T}) where {T <: Real}
    function out(x)
        inverse_metric(o,x)
    end
    out
end

# result in frame of t
# TODO: Write test for this
function toroidal_split(t::ToroidalEntry{T}, p) where {T <: Real}
    ploc = t.orthoframe' * (p - t.centre)
    base = [ploc[1:end-1] ; zero(T)]
    scaled_base = if iszero(base)
        [t.major_radius, zero(T), zero(T)]
    else
        normalize(base) .* t.major_radius
    end
    delta = ploc .- scaled_base
    (scaled_base, delta)
end

# purely positional transition
# TODO: write a simpler version of this to test against
function construct_torus_transition(a::ToroidalEntry{T}, b::ToroidalEntry{T}) where {T <: Real}
    funny_power = log(b.inner_minor_radius/b.outer_minor_radius)/log(a.outer_minor_radius/a.inner_minor_radius)
    rho_quotient = b.major_radius/a.major_radius
    function transition(p)
        sb, delta = toroidal_split(a, p)
        g = norm(delta)
        gb = b.outer_minor_radius * (g/a.inner_minor_radius)^funny_power
        delta_b = (gb/g) * delta
        pb_pre = rho_quotient .* sb .+ delta_b
        b.orthoframe * pb_pre + b.centre
    end
    transition
end

# TODO: remove magic constants
function construct_torus_throat_pair(a::ToroidalEntry{T}, b::ToroidalEntry{T}) where {T <: Real}
    EXIT_TOL = 0.1
    DT = 0.001
    OMEGA = 4.
    t_ab = OrientedThroat(a.ambient, b.ambient, a, b, construct_torus_transition(a, b), smooth_torus_parameter(a), nothing, construct_throat_ray_pusher(EXIT_TOL, DT, OMEGA))
    t_ba = OrientedThroat(b.ambient, a.ambient, b, a, construct_torus_transition(b, a), smooth_torus_parameter(b), t_ab, construct_throat_ray_pusher(EXIT_TOL, DT, OMEGA))
    t_ab.opposite = t_ba
    (t_ab, t_ba)
end

# modifies throat list in each ambient space
function construct_torus_throat_pair!(a::ToroidalEntry{T}, b::ToroidalEntry{T}) where {T <: Real}
    throat_pair = construct_torus_throat_pair(a,b)
    push!(a.ambient.throats, throat_pair[1])
    push!(b.ambient.throats, throat_pair[2])
    throat_pair
end

# assumption: input ray has euclidean space as ambient space
# maybe I should encode that at type level or something
# but really the ambient space types are too varied for that to be reasonable
function construct_euclidean_ray_pusher(entry_tol::T, skybox_norm::T) where {T <: Real}
    function pusher(r::Ray{T}, niter)
        ambient::Ambient{T} = r.spot
        q = deepcopy(r.q)
        last_i = 0
        ix = 0
        below_tol = false
        escaped = false
        for i in 1:niter
            last_i = i
            minsdf, ix = findmin(throat->sdf(throat.entry_a, q), ambient.throats)
            below_tol = minsdf < entry_tol
            escaped = norm(q) > skybox_norm
            if below_tol | escaped
                break
            else
                q .+= minsdf .* normalize(r.v)
            end
        end
        new_spot = if below_tol ambient.throats[ix] else ambient end
        new_iter_count = if escaped 0 else niter - last_i end
        Ray(q, r.v, new_spot), new_iter_count
    end
    pusher
end

# exit_tol must be limited by the minimal separation between throat entries in the ambient space
# incorrect results otherwise
# TODO: remove this dependency or make it easy to handle
# Also, need entry_tol < exit_tol, otherwise you get spurious terminations - i.e. the isosurface at which
# I enter the throat needs to be closer than the one at which I exit it.
function construct_throat_ray_pusher(exit_tol::T, dt::T, omega::T) where {T <: Real}
    function pusher(r::Ray{T}, niter)
        q = deepcopy(r.q)
        N = length(q)
        spot = r.spot
        im_q = inverse_metric(spot, q)
        p = im_q \ r.v
        x = deepcopy(q)
        y = deepcopy(p)
        im_jac_q = reshape(ForwardDiff.jacobian(inverse_metric(r.spot), q), (N,N,N))
    
        # just buffers
        im_x = zero(im_q)
        im_jac_x = zero(im_jac_q)
    
        TRANSITION_TOL = 0.2
        exited = false
        last_i = 0
        for i in 1:niter
            last_i = i
            tao!(inverse_metric(spot), q, p, x, y, im_q, im_jac_q, im_x, im_jac_x, dt, omega)
            if spot.partition(q) < TRANSITION_TOL
                new_spot = spot.opposite
    
                new_q = spot.transition(q)
                jac_q = ForwardDiff.jacobian(new_spot.transition, new_q)
                new_p = jac_q' * p
    
                new_x = spot.transition(x)
                jac_x = ForwardDiff.jacobian(new_spot.transition, new_x)
                new_y = jac_x' * y
    
                spot = new_spot
                q .= new_q
                p .= new_p
                x .= new_x
                y .= new_y
            end
            exited = sdf(spot.entry_a, q) > exit_tol
            if exited
                break
            end
        end
        im_q = inverse_metric(spot, q)
        v = im_q * p
        new_spot = if exited spot.ambient_a else spot end
        Ray(q, v, new_spot), niter - last_i
    end
    pusher
end

function push_ray(r::Ray{T}, niter) where {T <: Real}
    r.spot.ray_pusher(r, niter)
end

# NOTE: Invalidates im_x and im_jac_x. These do not need to be valid as inputs either - they're just buffers.
function tao!(inverse_metric, q, p, x, y, im_q, im_jac_q, im_x, im_jac_x, dt, omega)
    N = length(q)
    # step 1
    for i in 1:N
        p[i] -= (dt/4)*dot(y,(@view im_jac_q[:,:,i]),y)
    end
    x .+= (dt/2).*(im_q*y)
    # step 2
    im_jac_x .= reshape(ForwardDiff.jacobian(inverse_metric, x), (N,N,N))
    im_x .= inverse_metric(x)
    for i in 1:N
        y[i] -= (dt/4)*dot(p,(@view im_jac_x[:,:,i]),p)
    end
    q .+= (dt/2).*(im_x*p)
    # step 3
    possum = q + x
    posdif = q - x
    momsum = p + y
    momdif = p - y
    urot = cos(2*omega*dt)*posdif + sin(2*omega*dt)*momdif
    lrot = -sin(2*omega*dt)*posdif + cos(2*omega*dt)*momdif
    q .= (1/2).*(possum .+ urot)
    p .= (1/2).*(momsum .+ lrot)
    x .= (1/2).*(possum .- urot)
    y .= (1/2).*(momsum .- lrot)
    # step 4
    im_jac_x .= reshape(ForwardDiff.jacobian(inverse_metric, x), (N,N,N))
    im_x .= inverse_metric(x)
    for i in 1:N
        y[i] -= (dt/4)*dot(p,(@view im_jac_x[:,:,i]),p)
    end
    q .+= (dt/2).*(im_x*p)
    # step 5
    im_jac_q .= reshape(ForwardDiff.jacobian(inverse_metric, q), (N,N,N))
    im_q .= inverse_metric(q)
    for i in 1:N
        p[i] -= (dt/4)*dot(y,(@view im_jac_q[:,:,i]),y)
    end
    x .+= (dt/2).*(im_q*y)
end

function main_test()
    iddy = Matrix{Float64}(I, 3, 3)
    project_root = project_root_dir()
    # x, nx, y, ny, z, nz
    # bottom, top, right, left, front, back
    facenames = ["bottom", "top", "right", "left", "front", "back"]
    textures_1 = [load(joinpath(project_root, "skybox-" * facename * ".png")) for facename in facenames]
    skybox_texture_1 = SkyBoxTexture(textures_1...)
    skybox_1 = SkyBox(skybox_texture_1)
    textures_2 = [load(joinpath(project_root, "skybox2-" * facename * ".png")) for facename in facenames]
    skybox_texture_2 = SkyBoxTexture(textures_2...)
    skybox_2 = SkyBox(skybox_texture_2)

    # Elsewhere in file, EXIT_TOL must be larger (0.1 currently)
    ENTRY_TOL = 0.05
    SKYBOX_NORM = 20.

    ambient_1 = Ambient(OrientedThroat{Float64}[], skybox_1, euclidean_inverse_metric, construct_euclidean_ray_pusher(ENTRY_TOL, SKYBOX_NORM))
    ambient_2 = Ambient(OrientedThroat{Float64}[], skybox_2, euclidean_inverse_metric, construct_euclidean_ray_pusher(ENTRY_TOL, SKYBOX_NORM))

    torus_1 = ToroidalEntry(zeros(3), iddy, 1.0, 0.5, 0.25, ambient_1)
    torus_2 = ToroidalEntry(zeros(3), iddy, 1.0, 0.5, 0.25, ambient_2)

    # mutates the ambients
    ota, otb = construct_torus_throat_pair!(torus_1, torus_2)

    height = 512
    width = 512

    hoz_fov = pi/3
    z = (width/2) * cot(hoz_fov/2)
    camera_transform = RotX(pi/6) * RotY(-pi/4)
    camera_frame = camera_transform * [[1.,0.,0.] [0.,1.,0.] [0.,0.,z]]
    camera_centre = Vector(camera_transform * [0.,0.,-4.])
    camera = Camera(camera_centre, camera_frame, width, height, ambient_1)

    out = Matrix{RGB{N0f8}}(undef, height, width)

    for x in 1:height
        for y in 1:width
            niter = 6400
            @show x,y
            ray = pixel_to_ray(camera, x, y)
            # have to uniformize ray pushing for sure
            while niter > 0
                ray, niter = push_ray(ray, niter)
            end
            if norm(ray.q) > SKYBOX_NORM
                out[x,y] = get_ray_color(ray)
            else
                out[x,y] = RGB(1.,0.,0.)
            end
        end
    end
    out
end

# Should a ray be able to say it's "in an entry"?
# No, let's fix ray coordinates to ambient space.

# Let's say the two types of thing responsible for pushing rays are oriented throats
# and ambient spaces.

# so it turns out that we probably need, for each entry, a function
# that takes a ray in the ambient space and tells us whether that ray enters
# the entry eventually. More specifically, it should either tell us one of "yes, it enters [here]",
# "inconclusive, check back after evolving ray for another [x distance]" or "no, misses completely"
# sdfs are capable of this.
# What is the most important characteristic of an sdf anyway? Probably quasiconvexity?
# I kinda need some way of telling whether I might have "overshot".

# It seems like the actual criterion is being Lipschitz. If the constant is K, and my value at x is f(x),
# then I can step forward by f(x)/K, and may *just* hit zero thereby.
# Mellowmax is nonexpansive, so its Lipschitz constant is 1, so that will work just fine.


# I probably also want a way of saying "yeah you're completely past it". A simple sdf-analogue
# will not accomplish this. Maybe give each thing a bounding sphere?
# For now, I'll just solve this by saying that a ray has escaped when its norm becomes large enough.

# I need to come up with a more uniform ray pushing behavior. Part of the trouble is that *somewhere*
# there needs to be information about accuracy parameters, and this isn't uniform across pushers.
# Maybe a pusher should be an object itself? Something that composes local pushers?
# Eh. For now, I'll just bake the parameters into each pushing function, and bake each pushing function
# into an environment.

# Okay, but what should a pusher do, exactly? What's the interface?
# The generic pusher takes in a ray and a number of iterations, and spends at most that many iterations
# pushing the ray, but may spend fewer.

# Is the pusher responsible for transitioning the ray? Probably should be, yeah.

end