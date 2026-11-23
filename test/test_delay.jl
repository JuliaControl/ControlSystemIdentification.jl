using ControlSystemIdentification, ControlSystemsBase, Plots, Random, Statistics

Random.seed!(1)
Ts = 0.1
τ_samples = 20
P = c2d(tf(1, [1, 0.5, 1]) * delay(τ_samples * Ts), Ts)
u = randn(1, 400)
y = lsim(P, u)[1]
d = iddata(y, u, Ts)

# The input delay convention of `arx` is that the default `inputdelay = 1` is a delay-free strictly proper system
for metric in (:simulation, :prediction)
    sweep = find_delay(d, 2, 2, 0:40; metric)
    @test sweep.delay == τ_samples + 1
    @test sweep.delays == 0:40
    @test length(sweep.loss) == 41
    @test sweep.loss[sweep.delay+1] < 1e-6
    @test sweep.Ts == Ts
end

# separate validation data and noise
mk(N) = (u = randn(1, N); iddata(lsim(P, u)[1] .+ 0.01randn(1, N), u, Ts))
dt, dv = mk(600), mk(600)
sweep = find_delay(dt, 4, 4, 0:40; dval = dv)
@test abs(sweep.delay - (τ_samples + 1)) <= 2

# keyword arguments are forwarded to arx
sweep = find_delay(d, 2, 2, 15:25; λ = 1e-10)
@test sweep.delay == τ_samples + 1

# plotting and display
plot(sweep)
show(IOBuffer(), MIME("text/plain"), sweep)

# error handling
@test_throws ArgumentError find_delay(d, 2, 2, 0:40; metric = :foo)
@test_throws ArgumentError find_delay(d, 2, 2, 1:0)
@test_throws ArgumentError find_delay(iddata(randn(2, 50), randn(1, 50), 1), 2, 2)
