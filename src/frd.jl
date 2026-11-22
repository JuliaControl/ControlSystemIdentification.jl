import ControlSystemsBase: feedback

"""
    FRD(w, r)

Represents frequency-response data. `w` holds the frequency vector and `r` the response. Methods defined on this type include
- `+,-,*`
- `length, vec, sqrt`
- `plot`
- `feedback`
- [`freqvec`](@ref)
- [`tfest`](@ref) to estimate a rational model
- Indexing in the frequency domain using, e.g., `G[1Hz : 5Hz]`, `G[1rad : 5rad]`

If `r` represents a MIMO frequency response, the dimensions are `ny × nu × nω`.

An object `frd::FRD` can be plotted using `plot(frd, hz=false)` if `using Plots` has been called.
"""
struct FRD{WT<:AbstractVector{<:Real},RT<:AbstractArray} <: LTISystem{Continuous}
    w::WT
    r::RT
end
"Represents frequencies in Herz for indexing of `FRD` objects: `frd[2Hz:10Hz]`"
struct Hz <: Number
    i::Any
end
Base.:*(i, ::Type{Hz}) = Hz(i)

"Represents frequencies in rad/s for indexing of `FRD` objects: `frd[2rad:10rad]`"
struct rad <: Number
    i::Any
end
Base.:*(i, ::Type{rad}) = rad(i)
(::Colon)(start::Union{Hz,rad}, stop::Union{Hz,rad}) = (start, stop)

import Base: +, -, *, /, length, sqrt, getindex

"""
    FRD(w, sys::LTISystem)

Generate a frequency-response data object by evaluating the frequency response of `sys` at frequencies `w`.
"""
function FRD(w, s::LTISystem)
    if ControlSystemsBase.issiso(s)
        FRD(w, freqrespv(s, w))
    else
        FRD(w, freqresp(s, w))
    end
end
Base.vec(f::FRD) = f.r
*(f::FRD, f2)    = FRD(f.w, f.r .* vec(f2))
+(f::FRD, f2)    = FRD(f.w, f.r .+ vec(f2))
-(f::FRD, f2)    = FRD(f.w, f.r .- vec(f2))
/(f::FRD, f2)    = FRD(f.w, f.r ./ vec(f2))

# Below are required for ambiguities
*(f::FRD, f2::FRD) = FRD(f.w, f.r .* f2.r)
+(f::FRD, f2::FRD) = FRD(f.w, f.r .+ f2.r)
-(f::FRD, f2::FRD) = FRD(f.w, f.r .- f2.r)
/(f::FRD, f2::FRD) = FRD(f.w, f.r ./ f2.r)

*(f::FRD, f2::LTISystem) = *(f, FRD(f.w, f2))
+(f::FRD, f2::LTISystem) = +(f, FRD(f.w, f2))
-(f::FRD, f2::LTISystem) = -(f, FRD(f.w, f2))
/(f::FRD, f2::LTISystem) = /(f, FRD(f.w, f2))
/(f::FRD, f2::TransferFunction) = /(f, FRD(f.w, f2)) # Required for ambiguity

# Other direction
*(f::LTISystem, f2::FRD) = *(FRD(f2.w, f), f2)
+(f::LTISystem, f2::FRD) = +(FRD(f2.w, f), f2)
-(f::LTISystem, f2::FRD) = -(FRD(f2.w, f), f2)
/(f::LTISystem, f2::FRD) = /(FRD(f2.w, f), f2)

-(f::FRD) = FRD(f.w, -f.r)
length(f::FRD) = length(f.w)
function Base.size(f::FRD)
    noutputs(f), ninputs(f) # Size in the ControlSystemsBase sense
end
function Base.size(f::FRD, i::Int)
    if i == 1
        noutputs(f)
    elseif i == 2
        ninputs(f)
    else
        throw(ArgumentError("Index out of bounds"))
    end
end

# Base.convert(::Type{FRD}, sys::LTIsystem)  # This is not defined since the type does not store the frequency vector, and there is thus no way to convert to this type. Instead, arithmetic operations are defined since these take values rather than types

Base.lastindex(f::FRD) = length(f)
function Base.getproperty(f::FRD, s::Symbol)
    s === :Ts && return π / maximum(f.w)
    s === :timeevol && return Continuous()
    s === :nu && return ninputs(f)
    s === :ny && return noutputs(f)
    getfield(f, s)
end
Base.propertynames(f::FRD, private::Bool = false) = (fieldnames(typeof(f))..., :Ts, :timeevol, :nu, :ny)

function Base.show(io::IO, frd::FRD)
    write(io, "Frequency (rad/s)\n")
    write(io, "----------------\n")
    show(io, MIME("text/plain"), frd.w)
    write(io, "\n\n")
    write(io, "Response\n")
    write(io, "--------\n")
    show(io, MIME("text/plain"), frd.r)
end

ControlSystemsBase.noutputs(f::FRD) = f.r isa AbstractVector ? 1 : size(f.r, 1)
ControlSystemsBase.ninputs(f::FRD) = f.r isa AbstractVector ? 1 : size(f.r, 2)
ControlSystemsBase._default_freq_vector(f::Vector{<:FRD}, ::Any; kwargs...) = f[1].w
ControlSystemsBase._default_freq_vector(f::FRD, ::Any; kwargs...) = f.w
ControlSystemsBase.numeric_type(::Type{<:FRD{<:Any,RT}}) where RT = eltype(RT) # Used by, e.g., the nyquistplot recipe. FRD is not rational, so the recipe computes no poles or zeros
function ControlSystemsBase.bode(f::FRD, w::AbstractVector = f.w; unwrap=true, kwargs...) # kwargs such as balance do not apply to frequency-response data
    w == f.w || error("Frequency vector must match the one stored in the FRD")
    angles = angle.(f.r)
    angles = reshape(angles, f.ny, f.nu, :)
    unwrap && ControlSystemsBase.unwrap!(angles,3)
    @. angles = rad2deg(angles)
    reshape(abs.(f.r), f.ny, f.nu, :), angles, f.w
end

function ControlSystemsBase.freqresp(f::FRD, w::AbstractVector{W} = f.w; kwargs...) where W <: Real # kwargs such as balance do not apply to frequency-response data
    w == f.w || error("Frequency vector must match the one stored in the FRD")
    reshape(f.r, f.ny, f.nu, :)
end

sqrt(f::FRD) = FRD(f.w, sqrt.(f.r))
function getindex(f::FRD, i)
    if ControlSystemsBase.issiso(f)
        FRD(f.w[i], f.r[i])
    else
        FRD(f.w[i], f.r[:,:,i])
    end
end
getindex(f::FRD, i::Int) = f.r[i]
getindex(f::FRD, i::Int, j::Int) = (@assert(i == 1 && j == 1); f)
function getindex(f::FRD, r::Tuple{Hz,Hz})
    s = findfirst(2pi * r[1].i .< f.w)
    e = findlast(f.w .< 2pi * r[2].i)
    f[s:e]
end
function getindex(f::FRD, r::Tuple{rad,rad})
    s = findfirst(r[1].i .< f.w)
    e = findlast(f.w .< r[2].i)
    f[s:e]
end
getindex(f::Tuple{<:FRD, <:FRD}, r::Tuple) = (f[1][r], f[2][r])
Base.isapprox(f1::FRD, f2::FRD; kwargs...) =
    (f1.w == f2.w) && isapprox(f1.r, f2.r; kwargs...)
Base.:(==)(f1::FRD, f2::FRD) = (f1.w == f2.w) && ==(f1.r, f2.r)

function Base.hcat(f::FRD, f2::FRD...)
    all(f.w == f2.w for f2 in f2) || error("Frequency vectors must match")
    FRD(f.w, hcat(f.r, (f2.r for f2 in f2)...))
end

function Base.vcat(f::FRD, f2::FRD...)
    all(f.w == f2.w for f2 in f2) || error("Frequency vectors must match")
    FRD(f.w, vcat(f.r, (f2.r for f2 in f2)...))
end

sensitivity(P::FRD, K) = FRD(P.w, 1.0 ./ (1.0 .+ vec(P) .* vec(K)))
feedback(P::FRD, K) = FRD(P.w, vec(P) ./ (1.0 .+ vec(P) .* vec(K)))
feedback(P::FRD, K::Number=1) = FRD(P.w, vec(P) ./ (1.0 .+ vec(P) .* K))
feedback(P::Number, K::FRD) = FRD(K.w, P ./ (1.0 .+ P .* vec(K)))
feedback(P, K::FRD) = FRD(K.w, vec(P) ./ (1.0 .+ vec(P) .* vec(K)))
feedback(P::FRD, K::FRD) = feedback(P, vec(K))

feedback(P::FRD, K::LTISystem) = feedback(P, freqrespv(K, P.w))
feedback(P::LTISystem, K::FRD) = feedback(freqrespv(P, K.w), K)

"""
    freqvec(h, k)

Return a frequency vector of length `k` for systems with sample time `h`.
"""
freqvec(h, k::AbstractVector) = freqvec(h, length(k))
freqvec(h, k::Integer) = LinRange(0, π / h, k)

ControlSystemsBase.issiso(frd::FRD) = ndims(frd.r) == 1 || (size(frd.r, 1) == size(frd.r, 2) == 1)

"""
    c2d(w::AbstractVector{<:Real}, Ts; w_prewarp = 0)
    c2d(frd::FRD, Ts; w_prewarp = 0)

Transform continuous-time frequency vector `w` or frequency-response data `frd` from continuous to discrete time using a bilinear (Tustin) transform. This is useful in cases where a frequency response is obtained through frequency-response analysis, and the function [`subspaceid`](@ref) is to be used.
"""
function ControlSystemsBase.c2d(w::AbstractVector{<:Real}, Ts; w_prewarp=0)
    a = w_prewarp == 0 ? Ts/2 : tan(w_prewarp*Ts/2)/w_prewarp
    @. 2*atan(w*a)
end

ControlSystemsBase.c2d(f::FRD, Ts::Real; kwargs...) = FRD(c2d(f.w, Ts; kwargs...), f.r)


"""
    H, N = tfest(data, σ = 0.05, method = :corr)

Estimate a transfer function model using the Correlogram approach (default) using the signal model ``y = H(iω)u + n``.

Both `H` and `N` are of type `FRD` (frequency-response data).

- `σ` determines the width of the Gaussian window applied to the estimated correlation functions before FFT. A larger `σ` implies less smoothing.
- `H` = Syu/Suu             Process transfer function
- `N` = Sy - |Syu|²/Suu     Estimated Noise PSD (also an estimate of the variance of ``H``). Note that a PSD is related to the "noise model" ``N_m`` used in the system identification literature as ``N_{psd} = N_m^* N_m``. The magnitude curve of the noise model can be visualized by plotting `√(N)`.
- `method`: `:welch` or `:corr`. `:welch` uses the Welch method to estimate the power spectral density, while `:corr` (default) uses the Correlogram approach. If `method = :welch`, the additional keyword arguments `n`, `noverlap` and `window` determine the number of samples per segment (default 10% of data), the number of samples to overlap between segments (default 50%), and the window function to use (default `hamming`), respectively.

A Welch estimate pooled over several data segments is obtained by passing a vector of data objects, see `tfest(ds::AbstractVector{<:AbstractIdData})`.

# Extended help
This estimation method is unbiased if the input ``u`` is uncorrelated with the noise ``n``, but is otherwise biased (e.g., for identification in closed loop).
"""
function tfest(d::AbstractIdData, σ::Real = 0.05; method = :corr, n = length(d) ÷ 10, noverlap = n ÷ 2, window = hamming)
    d.nu == 1 || error("Cannot perform tfest on multiple-input data. Consider using time-domain estimation or statespace estimation.")
    if d.ny > 1
        HNs = [tfest(d[i,1], σ) for i in 1:d.ny]
        HR = reshape(reduce(vcat, [transpose(hn[1].r) for hn in HNs]), d.ny, 1, :)
        NR = reshape(reduce(vcat, [transpose(hn[2].r) for hn in HNs]), d.ny, 1, :)
        return FRD(HNs[1][1].w, HR), FRD(HNs[1][1].w, NR)
    end
    y, u, h = vec(output(d)), vec(input(d)), sampletime(d)
    if method === :welch
        Syy, Suu, Syu = wcfft(y, u, n = n, noverlap = noverlap, window = window)
    elseif method === :corr
        Syy, Suu, Syu = fft_corr(y, u, σ)
    else error("Unknown method $method") end
    return spectra2tfest(Syy, Suu, Syu, h, length(y))
end

"""
    H, N = tfest(ds::AbstractVector{<:AbstractIdData}; n = minimum(length, ds) ÷ 10, noverlap = n ÷ 2, window = hamming)

Estimate a transfer function using the Welch method, pooled over several data segments `ds` (for example, separate experiments or disjoint parts of one experiment). The segments may have different lengths, but must share the sample time and the numbers of inputs and outputs.

Every Welch window lies within a single segment. The cross- and auto-spectral sums are accumulated over the windows of all segments before `H` and `N` are formed, so that the result equals the Welch estimate over the union of the windows of all segments, rather than an average of per-segment estimates. A segment shorter than `n` contains no window and does not contribute. With a single segment, the result equals that of `tfest(ds[1]; method = :welch, n, noverlap, window)`.

The keyword arguments `n`, `noverlap` and `window` have the same meaning as for `tfest(d; method = :welch)`. The noise spectrum `N` is normalized by the total number of samples in the contributing segments. Only `method = :welch` is supported for multiple segments.

See also [`coherence`](@ref), which accepts a vector of data segments in the same way.
"""
function tfest(ds::AbstractVector{<:AbstractIdData}; method = :welch, n = minimum(length, ds) ÷ 10, noverlap = n ÷ 2, window = hamming)
    method === :welch || throw(ArgumentError("tfest with multiple data segments only supports method = :welch"))
    h, ny = check_segments(ds)
    if ny > 1
        HNs = [tfest([d[i,1] for d in ds]; n, noverlap, window) for i in 1:ny]
        HR = reshape(reduce(vcat, [transpose(hn[1].r) for hn in HNs]), ny, 1, :)
        NR = reshape(reduce(vcat, [transpose(hn[2].r) for hn in HNs]), ny, 1, :)
        return FRD(HNs[1][1].w, HR), FRD(HNs[1][1].w, NR)
    end
    ys = [vec(output(d)) for d in ds]
    us = [vec(input(d)) for d in ds]
    Syy, Suu, Syu = wcfft(ys, us; n, noverlap, window)
    L = sum(length(y) for y in ys if length(y) >= n)
    return spectra2tfest(Syy, Suu, Syu, h, L)
end

function spectra2tfest(Syy, Suu, Syu, h, L)
    w = freqvec(h, Syu)
    H = FRD(w, Syu ./ Suu)
    N = FRD(w, real.(Syy .- abs2.(Syu) ./ Suu) ./ L)
    return H, N
end

"""
    h, ny = check_segments(ds)

Verify that the data segments `ds` are non-empty, single-input, and share the sample time and the number of outputs. Returns the common sample time and number of outputs.
"""
function check_segments(ds::AbstractVector{<:AbstractIdData})
    isempty(ds) && throw(ArgumentError("At least one data segment is required"))
    allequal(sampletime(d) for d in ds) || throw(ArgumentError("All data segments must have the same sample time"))
    allequal(noutputs(d) for d in ds) || throw(ArgumentError("All data segments must have the same number of outputs"))
    all(ninputs(d) == 1 for d in ds) || throw(ArgumentError("Spectral estimation with multiple data segments only supports single-input data"))
    sampletime(ds[1]), noutputs(ds[1])
end

function fft_corr(y, u, σ = 0.05)
    n = length(y)
    w = gaussian(2n - 1, σ)

    Syu = rfft(ifftshift(w .* xcorr(y, u)))
    Syy = rfft(ifftshift(w .* xcorr(y, y)))
    Suu = rfft(ifftshift(w .* xcorr(u, u)))
    Syy, Suu, Syu
end



"""
    κ² = coherence(d; n = length(d)÷10, noverlap = n÷2, window=hamming, method=:welch)

Calculates the magnitude-squared coherence Function. κ² close to 1 indicates a good explainability of energy in the output signal by energy in the input signal. κ² << 1 indicates that either the system is nonlinear, or a strong noise contributes to the output energy.

- κ: Squared coherence function in the form of an [`FRD`](@ref).
- `method`: `:welch` or `:corr`. `:welch` uses the Welch method to estimate the power spectral density, while `:corr` uses the Correlogram approach . For `method = :corr`, the additional keyword argument `σ` determines the width of the Gaussian window applied to the estimated correlation functions before FFT. A larger `σ` implies less smoothing.

A Welch estimate pooled over several data segments is obtained by passing a vector of data objects, see `coherence(ds::AbstractVector{<:AbstractIdData})`.

See also [`coherenceplot`](@ref)

# Extended help:
For the signal model ``y = Gu + v``, ``κ²`` is defined as 
```math
κ(ω)^2 = \\dfrac{S_{uy}}{S_{uu} S_{yy}} = \\dfrac{|G(iω)|^2S_{uu}^2}{S_{uu} (|G(iω)|^2S_{uu}^2 + S_{vv})} = \\dfrac{1}{1 + \\dfrac{S_{vv}}{S_{uu}|G(iω)|^2}}
```
from which it is obvious that ``0 ≤ κ² ≤ 1`` and that κ² is close to 1 if the noise energy ``S_{vv}`` is small compared to the output energy due to the input ``S_{uu}|G(iω)|^2``.
"""
function coherence(d::AbstractIdData; n = length(d) ÷ 10, noverlap = n ÷ 2, window = hamming, method=:welch, σ = 0.05)
    noutputs(d) == 1 || throw(ArgumentError("coherence only supports a single output. Index the data object like `d[i,:]` to obtain the `i`:th output."))
    if ninputs(d) > 1
        return multiple_coherence(d; n, noverlap, window, σ)
    end
    y, u, h = vec(output(d)), vec(input(d)), sampletime(d)
    if method === :welch
        Syy, Suu, Syu = wcfft(y, u, n = n, noverlap = noverlap, window = window)
    else
        Syy0, Suu0, Syu = fft_corr(y, u, σ)
        Syy = real(Syy0)
        Suu = real(Suu0)
    end
    return spectra2coherence(Syy, Suu, Syu, h)
end

"""
    κ² = coherence(ds::AbstractVector{<:AbstractIdData}; n = minimum(length, ds) ÷ 10, noverlap = n ÷ 2, window = hamming)

Calculate the magnitude-squared coherence function using the Welch method, pooled over several data segments `ds` of possibly different lengths. The cross- and auto-spectral sums are accumulated over the Welch windows of all segments, each window lying within a single segment, before the coherence is formed. A segment shorter than `n` does not contribute. With a single segment, the result equals that of `coherence(ds[1]; n, noverlap, window)`.

Only single-input, single-output data and `method = :welch` are supported for multiple segments. See [`tfest`](@ref) for the corresponding transfer-function estimate.
"""
function coherence(ds::AbstractVector{<:AbstractIdData}; method = :welch, n = minimum(length, ds) ÷ 10, noverlap = n ÷ 2, window = hamming)
    method === :welch || throw(ArgumentError("coherence with multiple data segments only supports method = :welch"))
    h, ny = check_segments(ds)
    ny == 1 || throw(ArgumentError("coherence only supports a single output. Index the data objects like `d[i,:]` to obtain the `i`:th output."))
    Syy, Suu, Syu = wcfft([vec(output(d)) for d in ds], [vec(input(d)) for d in ds]; n, noverlap, window)
    return spectra2coherence(Syy, Suu, Syu, h)
end

function spectra2coherence(Syy, Suu, Syu, h)
    k = @. abs2(Syu) / (Suu * Syy)
    FRD(freqvec(h, k), k)
end

"""
    Syy, Suu, Syu = wcfft(y, u; n = length(y) ÷ 10, noverlap = n ÷ 2, window = hamming)
    Syy, Suu, Syu = wcfft(ys::AbstractVector{<:AbstractVector}, us::AbstractVector{<:AbstractVector}; n, noverlap, window)

Sums of the windowed periodograms (Welch method) of the output `y` and input `u`: the auto-spectral sums `Syy`, `Suu` and the cross-spectral sum `Syu`, over the non-negative frequency bins. The sums are not normalized by the number of windows.

If vectors of signal segments `ys, us` are given, the sums are accumulated over the windows of all segments, with every window lying within a single segment. A segment shorter than `n` contributes no window.
"""
wcfft(y::AbstractVector{<:Number}, u::AbstractVector{<:Number}; n = length(y) ÷ 10, kwargs...) =
    wcfft([y], [u]; n, kwargs...)

function wcfft(ys::AbstractVector{<:AbstractVector}, us::AbstractVector{<:AbstractVector}; n = minimum(length, ys) ÷ 10, noverlap = n ÷ 2, window = hamming)
    length(ys) == length(us) || throw(DimensionMismatch("The numbers of output and input segments differ"))
    win, norm2 = DSP.Periodograms.compute_window(window, n)
    nfft = nextfastfft(n)
    nf = nfft ÷ 2 + 1
    Syy = zeros(nf)
    Suu = zeros(nf)
    Syu = zeros(ComplexF64, nf)
    nwin = 0
    for (y, u) in zip(ys, us)
        length(y) == length(u) || throw(DimensionMismatch("Input and output segments have different lengths"))
        length(y) >= n || continue
        uw = arraysplit(u, n, noverlap, nfft, win)
        yw = arraysplit(y, n, noverlap, nfft, win)
        for i in eachindex(uw)
            xy = rfft(yw[i])
            xu = rfft(uw[i])
            @. Syu += xy * conj(xu)
            @. Syy += abs2(xy)
            @. Suu += abs2(xu)
        end
        nwin += length(uw)
    end
    nwin > 0 || throw(ArgumentError("No data segment is at least as long as the window length n = $n"))
    Syy, Suu, Syu
end

function multiple_coherence(d::AbstractIdData; n = length(d) ÷ 10, noverlap = n ÷ 2, window = hamming, σ = 0.05)
    noutputs(d) == 1 || throw(ArgumentError("multiple_coherence only supports a single output."))
    ninputs(d) > 1   || throw(ArgumentError("multiple_coherence requires multiple inputs. Use `coherence` for single input."))
    
    y = vec(output(d))
    u = time1(input(d))
    h = sampletime(d)

    Syy, Suu, Syu = wcfft_mimo(y, u, n = n, noverlap = noverlap, window = window)

    n_freqs = length(Syy)
    γ2 = zeros(n_freqs)
    
    for i in 1:n_freqs
        s_xy = conj(Syu[i]) 
        num = real(dot(s_xy, Suu[i] \ s_xy))
        γ2[i] = num / Syy[i]
    end

    γ2 = clamp.(γ2, 0.0, 1.0)

    Sch = FRD(freqvec(h, γ2), γ2)
    return Sch
end

function wcfft_mimo(y::AbstractVector, u::AbstractMatrix; n = length(y) ÷ 10, noverlap = n ÷ 2, window = hamming)
    n_samples, n_inputs = size(u)
    n_samples == length(y) || throw(DimensionMismatch("Input and output lengths differ"))

    win, norm2 = DSP.Periodograms.compute_window(window, n)
    
    # Dimensions for FFT
    n_freqs = n ÷ 2 + 1

    # Initialize Accumulators
    Syy = zeros(Float64, n_freqs)
    Suu = [zeros(ComplexF64, n_inputs, n_inputs) for _ in 1:n_freqs]
    Syu = [zeros(ComplexF64, n_inputs) for _ in 1:n_freqs] # Vector of vectors

    # Calculate window stride
    step = n - noverlap
    
    # Segment Loop (Manual splitting to handle Matrix u safely)
    for i in 1:step:(n_samples - n + 1)
        # Extract segments and apply window
        y_seg = y[i:i+n-1] .* win
        u_seg = u[i:i+n-1, :] .* win # Broadcast window across columns

        # Compute FFTs
        Y_f = rfft(y_seg)
        U_f = rfft(u_seg) # FFT along time dimension (dim 1)
        
        # Accumulate Spectra per frequency bin
        for k in 1:n_freqs
            Y_k = Y_f[k]       # Scalar
            U_k = U_f[k, :]    # Vector (1 x Inputs)

            # Auto-spectrum Output (Scalar)
            Syy[k] += abs2(Y_k)

            # Auto-spectral Matrix Input (Matrix: Inputs x Inputs)
            # Outer product: U * U'
            Suu[k] .+= U_k * U_k' 

            # Cross-spectral Vector (Vector: Inputs)
            # We compute S_yx = Y * U' (Output * conj(Input))
            Syu[k] .+= Y_k .* conj.(U_k)
        end
    end

    return Syy, Suu, Syu
end



"""
    ir, t, Σ = impulseest(d::AbstractIdData, n; λ=0, estimator=ls)

Estimates the system impulse response by fitting an `n`:th order FIR model. Returns impulse-response estimate, time vector and covariance matrix.

This function only supports single-output data, use [`okid`](@ref) for multi-output data.

See also [`impulseestplot`](@ref) and [`okid`](@ref).
"""
function impulseest(d::AbstractIdData, n; λ = 0, estimator = ls, cov=true)
    d.ny == 1 || error("impulseest only supports single-output data, consider using the function okid instead")
    h, y, u = d.Ts, time1(output(d)), time1(input(d))
    N = min(size(u, 1), size(y, 1))
    @views yt, A = getARXregressor(y[1:N], u[1:N,:], 0, d.nu == 1 ? n : fill(n, d.nu); inputdelay = d.nu == 1 ? 0 : zeros(Int, d.nu))
    A .*= h # We adjust for the sample time here in order to get both ir and Σ adjusted correctly
    ir = estimator(A, yt, λ)
    t = range(0, length = n, step = h)
    Σ = cov ? parameter_covariance(yt, A, ir, λ) : nothing
    if d.nu > 1
        ir = reshape(ir, :, d.nu)
    end
    ir, t, Σ
end
