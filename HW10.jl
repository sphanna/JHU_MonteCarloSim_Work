using LinearAlgebra
using Distributions
using Plots
Plots.plotlyjs()

#steepest descent with known gradient
function SDalgorithm(g,θstart,ak,N)
    θ = fill(θstart,N+1)
    for k in 1:N
       θ[k+1] = θ[k] - ak[k]*g(θ[k])
    end
    return θ
end

#SA steepest descent with stochastic gradient estimate Yk
function SGalgorithm(Yk,θstart,ak,N)
    θ = fill(θstart,N+1)
    for k in 1:N
       Vk = rand(V(θ[k]))
       θ[k+1] = θ[k] - ak[k]*Yk(θ[k],Vk)
    end
    return θ
end

Yk(θk,Vk,Q,dQ) = Q(θk,Vk) * dlogpv(Vk,θk) + dQ(θk,Vk);

#algorithm setup
N = 1000
a(k) = 0.1/((50 + k)^0.501)
ak = a.(0:N-1)

begin #shared setup#
    dlogpv(V,θ) = [(θ[1] - V)/((θ[1]-1)*θ[1]),0]
    V(θ) = Bernoulli(θ[1])
end

begin #15.5 setup#
    Q5(θ,V) = θ[2]^2 + (1 - θ[1])*(θ[2]-V)
    L5(θ) = θ[2]^2 + θ[2] - θ[1] - θ[1]*θ[2] + θ[2]^2
    dQ5(θ,V) = [V-θ[2], 2*θ[2] + 1 - θ[1]]
    θ5star = [1/3,-1/3]
    θ5₀ = [1/2,1/2]
    Y5_ = (θ,V) -> Yk(θ,V,Q5,dQ5)
end

begin #Exercise 15.6 setup
    Q(θ,V) = (θ[2] - 10)^2 + 2*V*θ[1] - V
    dQ(θ,V) = [2*V, 2*(θ[2] - 10)]
    L(θ) = (θ[2] - 10)^2 + 2*θ[1]^2 - θ[1]
    g(θ) = [4*θ[1] - 1, 2*(θ[2]-10)]
    θstar = [1/4,10]
    θ₀ = [0.5,10.5]
    Y_ = (θ,V) -> Yk(θ,V,Q,dQ)
end

θ = [SGalgorithm(Y_,θ₀,ak,1000) for i in 1:5]
θterms = last.(θ)
dists = [norm(θterms[i] - θstar) for i in 1:5]
var(dists)

θ1 = θ[1]
L.(θ1[[1,11,101,1001]])
L(θstar)

xrange = 0:0.01:1
yrange = 9.6:0.01:10.6
data = [L([i,j]) for i∈xrange,j∈yrange]'

#heatmap(xrange,yrange,data, palette = :BrBG_7)
L(x,y) = L([x,y])
contour(xrange,yrange,L,legend = :right, title = "Three Stochastic Gradient Search Runs over L(θ)", fill = true, xlabel = "λ", ylabel = "β")
plot!(first.(θ[1]),last.(θ[1]), label = "Run 1")
plot!(first.(θ[2]),last.(θ[2]), label = "Run 2")
plot!(first.(θ[3]),last.(θ[3]), label = "Run 3")
scatter!([first(θstar)],[last(θstar)], label = "θ*")
