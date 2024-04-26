module CpuWormhole
using LinearAlgebra
using ForwardDiff
using InteractiveUtils

function cross(x,y)
    @assert length(x) == 3 && length(y) == 3
    [x[2]*y[3]-x[3]*y[2], x[3]*y[1]-x[1]*y[3], x[1]*y[2]-x[2]*y[1]]
end

struct SphericalThroat{T <: Real}
    centre::Vector{T}
    orthotransform::Matrix{T} # should be an orthogonal transform fixing the orientation of the throat
    rim_radius::T
    outer_throat_radius::T
    inner_throat_radius::T
    transition_line_radius::T
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

function throat_inv_metric(a::SphericalThroat{T}, b::SphericalThroat{T}, a_point) where {T <: Real}
    b_point = throat_transition(a, b, a_point)
    J = ForwardDiff.jacobian(x->throat_transition(b,a,x), b_point)
    smp = smooth_throat_parameter(a, a_point)
    (1-smp) * Matrix(1.0I, 3, 3) + smp * J * J'
end

function test_throat()
    a = SphericalThroat(zeros(3), Matrix{Float64}(I, 3, 3), 2.0, 1.0, 0.5, 0.6)
    b = SphericalThroat(zeros(3), Matrix{Float64}(I, 3, 3), 5.0, 4.0, 2.0, 2.1)
    a_point = [0.65, 0.0, 0.0]
    throat_inv_metric(a, b, a_point)
end

end # module CpuWormhole
