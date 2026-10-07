using ControlSystemIdentification, ControlSystemsBase, Plots

G = c2d(tf(1, [1,1,1]), 0.02)
u = randn(1, 100)
res = lsim(G, u, range(0, length=100, step=0.02))
d = iddata(res)

crosscorplot(d)
autocorplot(d.y, d.Ts)


plot(d)

predplot(ss(G), d)
simplot(ss(G), d)
simplot(d, ss(G))

specplot(d)
welchplot(d)

Gh,_ = tfest(d)
plot(Gh, plotphase=true)

plot(tfest(d))

# FRD in the ControlSystemsBase plot recipes, which pass keyword arguments such as balance and adaptive
w = exp10.(LinRange(-1, 1, 50))
frd = FRD(w, tf(1, [1, 1, 1]))
mag, phase, _ = bode(frd, w; unwrap=false, balance=true)
@test vec(mag) ≈ abs.(frd.r)
bodeplot(frd)
bodeplot([frd, frd])
nyquistplot(frd)
sigmaplot(frd)