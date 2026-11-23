"""
    DelaySweep

Result of [`find_delay`](@ref). Fields:
- `delay`: the delay (in samples, using the convention of the `inputdelay` keyword of [`arx`](@ref)) with the smallest loss.
- `delays`: the candidate delays.
- `loss`: the normalized RMS validation error for each candidate delay.
- `Ts`: the sample time of the data.

A `DelaySweep` can be plotted using `plot(sweep)`.
"""
struct DelaySweep
    delay::Int
    delays::Vector{Int}
    loss::Vector{Float64}
    Ts::Float64
end

function Base.show(io::IO, ::MIME"text/plain", s::DelaySweep)
    print(io, "DelaySweep: best inputdelay = $(s.delay) samples")
    s.Ts > 0 && print(io, " ($(s.delay * s.Ts) s)")
    i = findfirst(==(s.delay), s.delays)
    print(io, ", normalized RMS error $(s.loss[i])")
end

"""
    find_delay(d::InputOutputData, na, nb, delays = 1:20; dval = d, metric = :simulation, skip, kwargs...)

Estimate the input delay of a single-input single-output system by fitting one ARX model [`arx`](@ref)`(d, na, nb; inputdelay = k, kwargs...)` for each candidate delay `k` in `delays` and comparing the models on validation data.

The delay convention is the same as for the `inputdelay` keyword of [`arx`](@ref): `k = 1` is the default and corresponds to a system without any delay beyond the one sample that is required for causality of a strictly proper discrete-time system, `k = 0` corresponds to a direct term, and a plant with a pure delay of `n` samples has `k = n + 1`.

The orders `na` and `nb` determine the dynamics that can be represented by each candidate model. `nb` should not be much larger than the number of numerator coefficients of the delay-free system, since a large `nb` allows the estimator to compensate a too short delay with leading coefficients of the B polynomial, which flattens the loss as a function of the delay. `na` should be at least the order of the delay-free system, and somewhat larger if the data contains measurement noise, since [`arx`](@ref) is biased by output noise and a too small `na` results in a biased delay estimate. The estimate is therefore most accurate for data with a high signal-to-noise ratio, and should otherwise be considered approximate, in which case the estimated delay can be refined by comparing other model structures using the same validation data.

# Arguments:
- `d`: Identification data used for estimation.
- `na`, `nb`: Orders of the A and B polynomials.
- `delays`: Candidate delays in samples.
- `dval`: Validation data. It is recommended to provide data that was not used for estimation. Defaults to `d`.
- `metric`: Either `:simulation` (default) or `:prediction`. The simulation error is considerably more sensitive to a wrong delay than the one-step prediction error, but requires that `dval` starts from rest (or that `skip` is chosen to exclude the transient). The prediction error is computed from the measured outputs and is not affected by the initial state.
- `skip`: The number of initial samples of `dval` that are excluded from the loss. By default `0` for `:simulation` and the largest regressor length among the candidates for `:prediction`, such that all candidates are compared on the same samples.
- `kwargs`: Additional keyword arguments are passed to [`arx`](@ref), e.g., `λ` for regularization.

# Returns
A [`DelaySweep`](@ref) with the fields `delay`, `delays`, `loss`, `Ts`. Plot the loss as a function of the delay using `plot(sweep)`. Loss values larger than the keyword argument `lossmax` (default `10`) are clipped in the plot, which prevents unstable candidate models from dominating the scale.

The estimate is limited to integer multiples of the sample time. Candidate models for which the simulation diverges, or that are otherwise non-finite, are never selected.

# Example
```julia
using ControlSystemIdentification, ControlSystemsBase, Plots
Ts = 0.1
P  = c2d(tf(1, [1, 0.5, 1]) * delay(2.0), Ts)
u  = randn(1, 400)
d  = iddata(lsim(P, u))
sweep = find_delay(d, 2, 2, 0:40)
sweep.delay # 21 samples, i.e., 20 samples of pure delay plus the one sample required for causality
plot(sweep)
```
"""
function find_delay(
    d::InputOutputData,
    na::Integer,
    nb::Integer,
    delays = 1:20;
    dval::InputOutputData = d,
    metric::Symbol = :simulation,
    skip::Integer = -1,
    kwargs...,
)
    d.ny == 1 && d.nu == 1 || throw(ArgumentError("find_delay only supports single-input single-output data."))
    dval.ny == 1 && dval.nu == 1 || throw(ArgumentError("The validation data must be single-input single-output."))
    metric in (:simulation, :prediction) || throw(ArgumentError("metric must be :simulation or :prediction"))
    delays = collect(Int, delays)
    isempty(delays) && throw(ArgumentError("delays must not be empty"))
    if skip < 0
        skip = metric === :prediction ? max(na, nb + maximum(delays) - 1) : 0
    end
    skip < length(dval) || throw(ArgumentError("skip must be smaller than the length of the validation data"))
    y = vec(output(dval))
    yr = y[skip+1:end]
    scale = rms(yr .- mean(yr))
    scale > 0 || throw(ArgumentError("The validation output is constant."))
    loss = map(delays) do k
        G = arx(d, na, nb; inputdelay = k, kwargs...)
        yh = if metric === :prediction
            vec(predict(G, dval))
        else
            Gs = ss(G)
            vec(simulate(Gs, dval, zeros(Gs.nx)))
        end
        e = (y .- yh)[skip+1:end]
        sqrt(mean(abs2, e)) / scale
    end
    _, i = findmin(l -> isfinite(l) ? l : Inf, loss)
    DelaySweep(delays[i], delays, loss, d.Ts === nothing ? 0.0 : Float64(d.Ts))
end

@recipe function plot_delaysweep(s::DelaySweep; lossmax = 10)
    xguide --> "Input delay [samples]"
    yguide --> "Normalized RMS error"
    yscale --> :log10
    title --> "Delay sweep"
    label --> ""
    @series begin
        seriestype := :scatter
        s.delays, clamp.(s.loss, eps(), lossmax)
    end
    @series begin
        seriestype := :vline
        linestyle := :dash
        seriescolor := :black
        label := "Best delay"
        [s.delay]
    end
end
