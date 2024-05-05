module CpuWormhole
using LinearAlgebra
using ForwardDiff
using InteractiveUtils
using Images
using Rotations

const PROJECT_ROOT = pkgdir(@__MODULE__)

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

mutable struct SphericalThroat{T <: Real}
    name::String
    centre::Vector{T}
    orthotransform::Matrix{T} # should be an orthogonal transform fixing the orientation of the throat
    rim_radius::T
    outer_throat_radius::T
    inner_throat_radius::T
    transition_line_radius::T
    opposite::Union{Nothing, SphericalThroat}
    skybox::SkyBox # does not have to have the same real type actually, we use it for convenience
end

struct Camera{T <: Real}
    centre::Vector{T}
    frame::Matrix{T} # up to positive scalar
    width::Int
    height::Int
    spot::SphericalThroat{T}
end

struct Ray{T <: Real}
    q::Vector{T}
    p::Vector{T}
    spot::SphericalThroat{T}
end

function vec_quad_intersection(v::Vector{T}, q::PQuad{T}) where {T<:Real}
    quad_normal = cross(q.u, q.v)
    quad_constant = dot(quad_normal, q.corner)
    λ = quad_constant/dot(quad_normal, v)
    λ * v
end

# also checks visibility
function vec_quad_coordinates(v::Vector{T}, q::PQuad{T}) where {T <: Real}
    E = [q.u q.v]
    quad_normal = cross(q.u, q.v)
    quad_constant = dot(quad_normal, q.corner)
    λ = quad_constant/dot(quad_normal, v)
    intersec_coords = pinv(E) * (λ * v - q.corner)
    visible = λ > 0 && all(x -> 0 <= x <= 1, intersec_coords)
    intersec_coords, visible
end

function ray_velocity(r::Ray{T}) where {T<:Real}
    im = throat_inv_metric(r.spot, r.spot.opposite, r.q)
    im * r.p
end

# x is row, y is column
#=
The camera frame works like:
z, view direction
 \
  \_______y
  |
  | screen plane
 x|

The top-left corner of the screen in screenspace is not
at the origin of the camera frame.
=#
# x in [1,h], y in [1,w]
function pixel_to_vector(frame::Matrix{T}, w, h, x, y) where {T <: Real}
    normalize(frame * [x-(h+1)/2, y-(w+1)/2, one(T)])
end

function pixel_to_vector(camera::Camera{T}, x, y) where {T <: Real}
    pixel_to_vector(camera.frame, camera.width, camera.height, x, y)
end

# inv_metric_matrix should be in relation to the camera's spot and centre
function pixel_to_ray(camera::Camera{T}, metric_matrix::Matrix{T}, x, y) where {T <: Real}
    velocity = pixel_to_vector(camera, x, y)
    momentum = metric_matrix * velocity
    Ray(camera.centre, momentum, camera.spot)
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
    v = ray_velocity(r)
    get_vec_color(v, r.spot.skybox)
end

function camera_metric(camera::Camera{T}) where {T <: Real}
    inv(throat_inv_metric(camera.spot, camera.spot.opposite, camera.centre))
end

function evolve_ray(r::Ray{T}, dt, niter, omega) where {T <: Real}
    spot = r.spot
    ivm = throat_inv_metric(spot, spot.opposite)
    N = length(r.q)
    q = deepcopy(r.q)
    jac_q = ForwardDiff.jacobian(ivm, q)
    jac_q = reshape(jac_q, (N,N,N))
    inv_q = ivm(q)
    p = deepcopy(r.p)
    x = deepcopy(q)
    y = deepcopy(p)
    for it in 1:niter
        # step 1
        for i in 1:N
            p[i] -= (dt/4)*dot(y,(@view jac_q[:,:,i]),y)
        end
        x .+= (dt/2).*(inv_q*y)
        # step 2
        jac_x = ForwardDiff.jacobian(ivm, x)
        jac_x = reshape(jac_x, (N,N,N))
        inv_x = ivm(x)
        for i in 1:N
            y[i] -= (dt/4)*dot(p,(@view jac_x[:,:,i]),p)
        end
        q .+= (dt/2).*(inv_x*p)
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
        jac_x = ForwardDiff.jacobian(ivm, x)
        jac_x = reshape(jac_x, (N,N,N))
        inv_x = ivm(x)
        for i in 1:N
            y[i] -= (dt/4)*dot(p,(@view jac_x[:,:,i]),p)
        end
        q .+= (dt/2).*(inv_x*p)
        # step 5
        jac_q = ForwardDiff.jacobian(ivm, q)
        jac_q = reshape(jac_q, (N,N,N))
        inv_q = ivm(q)
        for i in 1:N
            p[i] -= (dt/4)*dot(y,(@view jac_q[:,:,i]),y)
        end
        x .+= (dt/2).*(inv_q*y)
        # Now swap to other side if necessary
        if past_transition(spot, q)
            new_q = throat_transition(spot, spot.opposite, q)
            new_x = throat_transition(spot, spot.opposite, x)
            backer = x->throat_transition(spot.opposite, spot, x)
            back_jac_q = ForwardDiff.jacobian(backer, new_q)
            back_jac_x = ForwardDiff.jacobian(backer, new_x)
            new_p = back_jac_q' * p
            new_y = back_jac_x' * y

            spot = spot.opposite
            q = new_q
            x = new_x
            p = new_p
            y = new_y

            ivm = throat_inv_metric(spot, spot.opposite)
            jac_q = ForwardDiff.jacobian(ivm, q)
            jac_q = reshape(jac_q, (N,N,N))
            inv_q = ivm(q)
        end
    end
    Ray(q, p, spot)
end

# maybe I should rewrite this one to be applicable to multiple points at once?
# partition-of-unity convention, 1 on the inside of the fiber and 0 on the outside
function throat_parameter(throat::SphericalThroat{T}, coord_point) where {T<:Real}
    delta = coord_point - throat.centre
    param = (throat.outer_throat_radius - norm(delta))/(throat.outer_throat_radius - throat.inner_throat_radius)
    param
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

function smooth_throat_parameter(throat::SphericalThroat{T}, coord_point) where {T <: Real}
    λ = throat_parameter(throat, coord_point)
    smootherstep(λ)
end

function throat_norm(throat::SphericalThroat{T}, param) where {T <: Real}
    (1-param) * throat.outer_throat_radius + param * throat.inner_throat_radius
end

# returns the b_point corresponding to a_point, if possible
function throat_transition(a::SphericalThroat{T}, b::SphericalThroat{T}, a_point) where {T <: Real}
    λ_a = throat_parameter(a, a_point)
    λ_b = (1-λ_a)
    delta_a = a_point - a.centre
    norm_ratio = throat_norm(b, λ_b)/norm(delta_a)
    delta_b = norm_ratio * (b.orthotransform * (a.orthotransform \ delta_a))
    delta_b + b.centre
end

function past_transition(a::SphericalThroat{T}, a_point) where {T <: Real}
    norm(a_point - a.centre) < a.transition_line_radius
end

# correct far from throat and also in the good transition region
function throat_inv_metric(a::SphericalThroat{T}, b::SphericalThroat{T}, a_point) where {T <: Real}
    b_point = throat_transition(a, b, a_point)
    J = ForwardDiff.jacobian(x->throat_transition(b,a,x), b_point)
    smp = smooth_throat_parameter(a, a_point)
    (1-smp) * Matrix(1.0I, 3, 3) + smp * J * J'
end

function throat_inv_metric(a::SphericalThroat{T}, b::SphericalThroat{T}) where {T <: Real}
    function out(x)
        throat_inv_metric(a,b,x)
    end
    out
end

# function test_throat()
#     a = SphericalThroat(zeros(3), Matrix{Float64}(I, 3, 3), 2.0, 1.0, 0.5, 0.6, nothing)
#     b = SphericalThroat(zeros(3), Matrix{Float64}(I, 3, 3), 5.0, 4.0, 2.0, 2.1, a)
#     a.opposite = b
#     a_point = [0.65, 0.0, 0.0]
#     throat_inv_metric(a, b, a_point)
# end

function project_root_dir()
    dir = @__DIR__
    while !ispath(joinpath(dir, "Project.toml"))
        dir = dirname(dir)
    end
    return dir
end

function just_do_some_shit()
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
    a = SphericalThroat("a",zeros(3), iddy, 2.0, 1.0, 0.5, 0.6, nothing, skybox_1)
    b = SphericalThroat("b",zeros(3), iddy, 2.0, 1.0, 0.5, 0.6, a, skybox_2)
    a.opposite = b

    height = 240
    width = 240

    hoz_fov = pi/3
    z = (width/2) * cot(hoz_fov/2)
    camera_frame = [[1.,0.,0.] [0.,1.,0.] [0.,0.,z]]
    camera_centre = [0.,0.,-4.]
    camera = Camera(camera_centre, camera_frame, width, height, a)
    camera_mm = camera_metric(camera)
    out = Matrix{RGB{N0f8}}(undef, height, width)
    niter = 400
    dt = 0.025
    omega = 4.
    for x in 1:height
        for y in 1:width
            @show x,y
            ray = pixel_to_ray(camera, camera_mm, x, y)
            @show ray.q
            @show ray.spot.name
            ray = evolve_ray(ray, dt, niter, omega)
            @show ray.q
            @show ray.spot.name
            out[x,y] = get_ray_color(ray)
        end
    end
    out
    # @show a
end

end # module CpuWormhole
