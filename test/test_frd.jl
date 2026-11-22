using ControlSystemIdentification, Random, ControlSystemsBase, Test, Statistics, Plots

Random.seed!(1)
##
T           = 100000
h           = 1
t           = range(0, step = h, length = T)
sim(sys, u) = lsim(c2d(sys, h), u, t)[1]
σy          = 0.5
sys         = tf(1, [1, 2 * 0.1, 0.1])
ωn          = sqrt(0.3)
sysn        = tf(σy * ωn, [1, 2 * 0.1 * ωn, ωn^2])

u  = randn(1,T)
y  = sim(sys, u)
yn = y + sim(sysn, randn(size(u)))
d  = iddata(y, u, h)
dn = iddata(yn, u, 1)

# using BenchmarkTools
# @btime begin
# Random.seed!(0)
k = coherence(d)
@test mean(k.r) > 0.98

k = coherence(dn)
kcorr = coherence(dn, method=:corr, σ=0.01)

@test all(k.r[1:10] .> 0.9)
@test k.r[end-1] > 0.5

@test all(kcorr.r[1:10] .> 0.9)
@test kcorr.r[end-1000] > 0.3 # This series is much longer than with Welch method

i = findfirst(k.w .> ωn)
@test mean(k.r[i .+ (-2:5)]) < 0.6
G, N = tfest(dn, 0.02)
G2, N2 = tfest(dn, method=:welch)
@test ControlSystemsBase.issiso(G) 
noisemodel = innovation_form(ss(sys), syse = ss(sysn))
noisemodel.D .*= 0
bodeplot(
    [sys, sysn],
    exp10.(range(-3, stop = log10(pi), length = 200)),
    layout    = (1, 4),
    plotphase = false,
    subplot   = [1, 2],
    size      = (3 * 800, 600),
    linecolor = :blue,
)#, ylims=(0.1,300))

coherenceplot!(dn, subplot = 3)
crosscorplot!(dn, -10:100, subplot = 4)
plot!(G, subplot = 1, lab = "G Est", alpha = 0.3, title = "Process model")
plot!(√N, subplot = 2, lab = "N Est", alpha = 0.3, title = "Noise model")

plot!(G2, subplot = 1, lab = "G Est W", alpha = 0.3, title = "Process model")
plot!(√N2, subplot = 2, lab = "N Est W", alpha = 0.3, title = "Noise model")



@testset "multiple coherence" begin
    @info "Testing multiple coherence"
    nu = 2
    nx = 2
    ny = 1
    N = 10000
    u = randn(nu, N)
    sys = ssrand(ny, nu, nx, Ts=1, proper=true)
    sys.B .= I(nu)
    y, t, x = lsim(sys, u)
    y .+= sin.(1 .* t')
    d = iddata(y, u, 1)
    c = coherence(d)
    plot(c, yscale=:identity)
    coherenceplot(d)

    @test mean(c.r) > 0.95
    using ControlSystemIdentification: rad
    @test minimum(c[0.99rad:1.01rad].r) < 0.2 # low coherence at disturbance frequency
end




for op in (+, -, *, /)
    @test op(G, G) isa FRD
end



ω = exp10.(LinRange(-2, 2, 100))
P, C = tf(1.0, [1, 1]), pid(1, 1)
S, D, N, T = gangoffour(P, C)
S2, D2, N2, T2 = gangoffour(FRD(ω, P), FRD(ω, C))

@test ninputs(S2) == noutputs(S2) == 1
@test size(S2) == (1, 1)
@test S2 == S2
@test FRD(ω, S) ≈ S2
@test FRD(ω, D) ≈ D2
@test FRD(ω, N) ≈ N2
@test FRD(ω, T) ≈ T2

@test FRD(ω, P)*P ≈ FRD(ω, P)*FRD(ω, P)
@test FRD(ω, P)-P ≈ FRD(ω, P)-FRD(ω, P)
@test FRD(ω, P)+P ≈ FRD(ω, P)+FRD(ω, P)
@test FRD(ω, P)/ss(P) ≈ FRD(ω, P)/FRD(ω, P) 
@test FRD(ω, P)/P ≈ FRD(ω, P)/FRD(ω, P)

P = ss(P)
@test FRD(ω, P)*P ≈ FRD(ω, P)*FRD(ω, P)
@test FRD(ω, P)-P ≈ FRD(ω, P)-FRD(ω, P)
@test FRD(ω, P)+P ≈ FRD(ω, P)+FRD(ω, P)
@test FRD(ω, P)/ss(P) ≈ FRD(ω, P)/FRD(ω, P) 
@test FRD(ω, P)/P ≈ FRD(ω, P)/FRD(ω, P)


## Algebra
G = tf(1, [1, 2])
w = exp10.(LinRange(-2, 2, 20))
Gf = FRD(w, G)

@test feedback(Gf, 1) ≈ FRD(w, feedback(G, 1))
@test feedback(1, Gf) ≈ FRD(w, feedback(1, G))

@test feedback(Gf, Gf) ≈ FRD(w, feedback(G, G))

@test feedback(Gf, G) ≈ FRD(w, feedback(G, G))
@test feedback(G, Gf) ≈ FRD(w, feedback(G, G))


## Indexing
Gf[0.011rad:0.03rad] == Gf[2:3]
Gf[(0.011/2pi)Hz:(0.03/2pi)Hz] == Gf[2:3]


## MIMO
G = ss(tf(1, [1,1,1]))
G = let (A,B,C,D) = ssdata(G)
    ss(A,B,I(2),0)
end
G = c2d(G, 0.01)


T = 1000
u = randn(1,T)
res = lsim(G, u)
d = iddata(res)
H,N = tfest(d)
@test size(H.r,1) == 2
@test size(H.r,2) == 1
@test size(H.r,3) == T

@test !ControlSystemsBase.issiso(H)
plot(H, plotphase=true)

@test H.Ts == G.Ts
H5 = H[1:5]
@test H5.w == H.w[1:5]
@test H5.r == H.r[:,:,1:5]


## MIMO
ny,nu,nx = 2,3,4                     # number of outputs, inputs and states
Ts = 1                                  # Sample time
G = ssrand(ny,nu,nx; Ts, proper=true);  # Generate a random system

N = 200                                 # Number of frequency points
w = range(1e-6, stop=pi/Ts-1/N, length=N)  # Frequency vector

frd = FRD(w, G);                        # Build a frequency-response data object
@test frd.r == freqresp(G, w)
@test frd.w == w
@test size(frd) == (ny, nu)

@test bode(frd) == bode(G, w)
@test nyquist(frd) == nyquist(G, w)
@test ControlSystemsBase.iscontinuous(frd)
@test freqresp(frd, w; balance = false) == freqresp(frd)

@test size([frd frd]) == (ny, 2nu)
@test size([frd; frd]) == (2ny, nu)
@test size([frd frd; frd frd]) == (2ny, 2nu)
@test size([frd frd], 1) == ny
@test size([frd frd], 2) == 2nu
@test_throws ArgumentError size([frd frd], 3)

# Tests for FRD printing
littlefrd = FRD([1, 2], [im, 2*im])
@test_nowarn show(littlefrd)
@testset "Welch estimation over multiple data segments" begin
    @info "Testing Welch estimation over multiple data segments"
    Random.seed!(0)
    G = tf([1, 0.5], [1, -0.9, 0.5], 1)
    nw = 256
    kw = (; n = nw, noverlap = nw ÷ 2, window = DSP.hanning)

    # A single segment gives the same result as the single-dataset method
    u1 = randn(1, 3000)
    d1 = iddata(lsim(G, u1)[1] .+ 0.1 .* randn(1, 3000), u1, 1)
    H1, N1 = tfest(d1; method = :welch, kw...)
    Hv, Nv = tfest([d1]; kw...)
    @test Hv.w == H1.w
    @test Hv.r ≈ H1.r
    @test Nv.r ≈ N1.r
    @test coherence([d1]; kw...).r ≈ coherence(d1; kw...).r
    @test tfest([d1])[1].r ≈ tfest(d1, method = :welch)[1].r # default window length
    @test coherence([d1]).r ≈ coherence(d1).r

    # Several segments of different lengths give the result of summing the Welch windows
    # of all segments; a segment shorter than the window contributes nothing
    lengths = [3000, 1700, 900, 2600, nw - 1]
    ds = map(lengths) do N
        u = randn(1, N)
        iddata(lsim(G, u)[1] .+ 0.2 .* randn(1, N), u, 1)
    end
    sums = [CSI.wcfft(vec(d.y), vec(d.u); kw...) for d in ds[1:end-1]]
    Syy = sum(s[1] for s in sums)
    Suu = sum(s[2] for s in sums)
    Syu = sum(s[3] for s in sums)
    Ryy, Ruu, Ryu = CSI.wcfft([vec(d.y) for d in ds], [vec(d.u) for d in ds]; kw...)
    @test Ryy ≈ Syy
    @test Ruu ≈ Suu
    @test Ryu ≈ Syu
    H, N = tfest(ds; kw...)
    @test H.r ≈ Syu ./ Suu
    @test N.r ≈ (Syy .- abs2.(Syu) ./ Suu) ./ sum(lengths[1:end-1])
    @test coherence(ds; kw...).r ≈ abs2.(Syu) ./ (Suu .* Syy)

    # A known transfer function is recovered from several noisy segments
    ds = map(1:20) do _
        N = rand(1000:4000)
        u = randn(1, N)
        iddata(lsim(G, u)[1] .+ 0.1 .* randn(1, N), u, 1)
    end
    H, N = tfest(ds; kw...)
    k = coherence(ds; kw...)
    band = findall(w -> 0 < w < 2, H.w) # the band where the output is dominated by the input
    Gtrue = vec(freqresp(G, H.w[band]))
    @test maximum(abs.(H.r[band] .- Gtrue) ./ abs.(Gtrue)) < 0.05
    @test minimum(k.r[band]) > 0.95

    # Multiple outputs are handled output by output
    dsm = [iddata([vec(d.y) vec(d.y)]', d.u, 1) for d in ds[1:3]]
    Hm, Nm = tfest(dsm; kw...)
    Hs, Ns = tfest(ds[1:3]; kw...)
    @test size(Hm.r) == (2, 1, length(Hs.w))
    @test Hm.r[2, 1, :] ≈ Hs.r

    # Errors
    @test_throws ArgumentError tfest(ds; method = :corr)
    @test_throws ArgumentError coherence(ds; method = :corr)
    @test_throws ArgumentError tfest([ds[1], iddata(ds[2].y, ds[2].u, 2)]; kw...)
    @test_throws ArgumentError tfest(ds[1:2]; n = 10^5)
end
