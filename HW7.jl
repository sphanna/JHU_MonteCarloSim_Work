using LinearAlgebra
using Plots
using Distributions
Plots.plotlyjs()

function tCI(x,conf_level=0.95)
    N = length(x)
    alpha = (1 - conf_level)
    tstar = quantile(TDist(N-1), 1 - alpha/2)
    r = tstar * std(x)/sqrt(N)
    s = mean(x)
    return [s - r, s + r]
end

function SPSA(y,θ₀,Δ,aₖ::AbstractArray,cₖ::AbstractArray,N)
    θ = fill(θ₀,N+1)
    θ[1] = θ₀
    p = length(θ₀)
    Δₖ = [Δ(p) for k in 1:N]
    [θ[k+1] = θ[k] - aₖ[k]*gSPSA(y,θ[k],Δₖ[k],cₖ[k]) for k in 1:N]
    return θ
end

function gSPSA(y,θₖ,Δ,cₖ)
    ckΔ = cₖ*Δ
    (1/(2*cₖ))*(y(θₖ+ckΔ) - y(θₖ-ckΔ)) * inv.(Δ)
end

t1 = 2^(1/3)
t2 = (1/2)*t1
t1 = factorial(10)^(1/11)
θstar = [(1/i)*t1 for i in 1:10]
θ₀ = 1.1*θstar

aₖ(k,a,A,α) = a/(k+1+A)^α
cₖ(k,c,γ) = c/(k+1)^γ

L(θ) = sum([i*θ[i] for i in 1:10]) + prod((1 ./ θ))
Lnorm(L,θ,θ₀,θstar) = (L(θ)-L(θstar))/(L(θ₀)-L(θstar))
y(θ,L,ϵ) = L(θ) + ϵ(θ) 
y_ = θ -> y(θ,L,ϵ_)

V = MvNormal(zeros(11),0.001^2*I)
ϵ(θ,V) = [θ...,1]'rand(V)
ϵ(θstar,V)
ϵ_ = θ -> ϵ(θ,V)

begin #setup
    N = 20000
    a = 0.01; A = 1000; c = 0.015; α = 0.602; γ = 0.101
    ak = aₖ.(1:N,a,A,α)
    ck = cₖ.(1:N,c,γ)
    Δ(p) = rand((-1,1),p)
    Nreps = 10
end

θterms =  [last(SPSA(y_,θ₀,Δ,ak,ck,N)) for i in 1:Nreps]
meanθterm = mean(θterms)

Lnorm(L,meanθterm,θ₀,θstar)

tCI(θterms,0.90)


L(θterm)
L(θstar)
L(ones(10))
L(θstar)