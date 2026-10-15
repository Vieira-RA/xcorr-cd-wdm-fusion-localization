"""Prototype: sub-sample delay bias of peak-interpolation methods.

Compares, on the same 2-D signals:
  parabolic          - current pipeline (signal_processing.parabolic_corr_delay_2d)
  upsampled          - parabolic on a U-times zero-padded cross-correlation
  gauss              - 5-point Gaussian (log-parabola) fit around the peak
Reports bias and std vs a known fractional delay, across SNR.
"""
import numpy as np
from signal_processing import parabolic_corr_delay_2d

U = 16


def corr2d(a, b, U=1):
    N = len(a)
    n = U * N
    f = np.fft.fftfreq(n, d=1.0 / U)          # lag index in samples
    Ax = np.fft.fft(a[:, 0], n); Bx = np.fft.fft(b[:, 0], n)
    Ay = np.fft.fft(a[:, 1], n); By = np.fft.fft(b[:, 1], n)
    cx = np.abs(np.fft.ifft(Ax * np.conj(Bx)))
    cy = np.abs(np.fft.ifft(Ay * np.conj(By)))
    c = np.fft.fftshift(cx + cy)
    lags = np.fft.fftshift(f)                 # in samples, spacing 1/U
    return lags, c


def parabola(y, k):
    yl, yc, yr = y[k - 1], y[k], y[k + 1]
    den = yl - 2 * yc + yr
    return 0.0 if abs(den) < 1e-15 else (yl - yr) / (2 * den)


def gauss_vertex(y, k, half=2):
    idx = np.arange(-half, half + 1)
    yy = np.log(np.clip(y[k - half:k + half + 1], 1e-300, None))
    c = np.polyfit(idx, yy, 2)
    return -c[1] / (2 * c[0]) if abs(c[0]) > 1e-18 else 0.0


def est_parabolic(a, b, dt):
    return parabolic_corr_delay_2d(a, b, dt)


def est_upsampled(a, b, dt, U=U):
    lags, c = corr2d(a, b, U)
    k = int(np.argmax(c))
    return (lags[k] + parabola(c, k) / U) * dt


def est_gauss(a, b, dt):
    lags, c = corr2d(a, b, 1)
    k = int(np.argmax(c))
    return (lags[k] + gauss_vertex(c, k)) * dt


def main():
    B = 22000.0
    fs = 20 * B
    dt = 1.0 / fs
    sig_t = 0.3748 / B
    t_end = 160 * sig_t
    n = int(t_end / dt) + 1
    t = np.linspace(0, t_end, n); t0 = t_end / 2
    pulse = np.exp(-0.5 * ((t - t0) / sig_t) ** 2)
    sig = np.column_stack([pulse, 0.3 * pulse])
    freq = np.fft.fftfreq(n, d=dt)
    true = 0.37 * dt                              # fractional sample delay
    shift = np.fft.ifft(np.fft.fft(sig, axis=0)
                        * np.exp(2j * np.pi * freq * true)[:, None], axis=0).real
    E = np.mean(np.sum(sig ** 2, axis=0))
    methods = [("parabolic", est_parabolic), ("upsampled", est_upsampled),
               ("gauss", est_gauss)]
    print("true delay = %.4f samples (dt=%.3g s)" % (true / dt, dt))
    print("%6s | %-11s %-11s %-11s" % ("SNR", *[m[0] for m in methods]))
    for snr in (40, 60, 80, 100, 120):
        s2 = E / 10 ** (snr / 10)
        rng = np.random.RandomState(0)
        res = {m[0]: np.empty(3000) for m in methods}
        for r in range(3000):
            a = sig + np.sqrt(s2) * rng.randn(*sig.shape)
            b = shift + np.sqrt(s2) * rng.randn(*shift.shape)
            for name, fn in methods:
                res[name][r] = fn(a, b, dt)
        line = "%6d |" % snr
        for name, _ in methods:
            bias = (np.mean(res[name]) - true) / dt
            std = np.std(res[name]) / dt
            line += " %+7.4f±%.4f" % (bias, std)
        print(line)
    print("(values in samples: bias +/- std)")


if __name__ == "__main__":
    main()
