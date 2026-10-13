# Cramér–Rao Bound (CRB) for Time-Delay Estimation — Simple Summary

## 1. Core definition
The CRB is a lower bound on the variance (or MSE for an unbiased estimator) of any unbiased estimator of a deterministic parameter.

For a delay \(\tau\):

\[
\operatorname{Var}(\hat{\tau}) \ge \operatorname{CRB}(\hat{\tau}) = \frac{1}{J(\tau)}
\]

where \(J(\tau)\) is the Fisher information:

\[
J(\tau) = -\mathbb{E}\left[\frac{\partial^2 \ln p(r|\tau)}{\partial \tau^2}\right]
\]

It is a **local / asymptotic** bound: valid for high SNR and/or long observation time. It does **not** capture threshold or ambiguity errors.

---

## 2. Simplest known deterministic signal + AWGN case
Model:

\[
r(t) = s(t-\tau) + n(t), \qquad 0 \le t \le T
\]

with additive white Gaussian noise (AWGN) having two-sided PSD \(N_0\).

Then:

\[
\boxed{
\operatorname{CRB}(\hat{\tau}) = \frac{1}{\mathrm{SNR}\,\beta^2}
}
\]

where:

\[
\boxed{
\mathrm{SNR} = \frac{E}{N_0}
}
\]

\[
E = \int_0^{T_s} s^2(t)\,dt
\]

and \(\beta^2\) is the **mean-square bandwidth**:

\[
\beta^2 =
\frac{\int_0^{T_s} \left(\frac{ds(t)}{dt}\right)^2 dt}
     {\int_0^{T_s} s^2(t)\,dt}
=
\frac{\int_{-\infty}^{\infty} (2\pi f)^2 |S(f)|^2 df}
     {\int_{-\infty}^{\infty} |S(f)|^2 df}
\]

So: larger SNR or larger bandwidth \(\Rightarrow\) smaller CRB.

---

## 3. SNR definitions to highlight

### 3.1 Deterministic known signal
\[
\boxed{
\mathrm{SNR} = \frac{E}{N_0}
}
\]

- \(E\): signal energy
- \(N_0\): two-sided noise power spectral density

**This SNR is a property of the received signal, not the cross-correlation.**

---

### 3.2 Two-sensor Gaussian random signal model
\[
r_1(t)=s(t)+n_1(t), \qquad
r_2(t)=s(t-\tau)+n_2(t)
\]

The effective SNR used in the CRB is:

\[
\boxed{
\mathrm{SNR}(\omega)=
\frac{
\left(\frac{S(\omega)}{N_1(\omega)}\right)
\left(\frac{S(\omega)}{N_2(\omega)}\right)
}{
1+\frac{S(\omega)}{N_1(\omega)}+\frac{S(\omega)}{N_2(\omega)}
}
}
\]

where \(S(\omega)\), \(N_1(\omega)\), \(N_2(\omega)\) are the PSDs of the signal and the two noises.

**Key point:** This is an *effective SNR* that combines the individual sensor SNRs. It is defined on the **separated received signals**, before any cross-correlation is performed.

---

### 3.3 Flat spectra, equal noise case
If \(S(\omega)=S\) and \(N_1(\omega)=N_2(\omega)=N_0\) over the band:

\[
\boxed{
\mathrm{SNR}=
\frac{(S/N_0)^2}{1+2(S/N_0)}
}
\]

This is the SNR that enters the two-sensor Fisher information:

\[
J_{CR}(\hat{\tau})=
\frac{T}{2\pi}\int_{-\infty}^{\infty}
\omega^2 \mathrm{SNR}(\omega)\,d\omega
\]

so:

\[
\operatorname{CRB}(\hat{\tau}) = 1/J_{CR}(\hat{\tau})
\]

---

## 4. SNR is related to the separated signals, not the cross-correlation signal

### 4.1 The SNR is defined on the received signals
In the two-sensor model:

\[
r_1(t) = s(t) + n_1(t)
\]
\[
r_2(t) = s(t-\tau) + n_2(t)
\]

The individual sensor SNRs are:

\[
S(\omega)/N_1(\omega)
\qquad\text{and}\qquad
S(\omega)/N_2(\omega)
\]

These are the ratios of the signal PSD to the noise PSD **at the output of each receiver**, before any correlation is performed.

### 4.2 The effective SNR combines the two separated signals
The quantity that actually enters the CRB formula is the effective SNR:

\[
\mathrm{SNR}(\omega)=
\frac{
\left(\frac{S(\omega)}{N_1(\omega)}\right)
\left(\frac{S(\omega)}{N_2(\omega)}\right)
}{
1+\frac{S(\omega)}{N_1(\omega)}+\frac{S(\omega)}{N_2(\omega)}
}
\]

This is **not** the SNR of the cross-correlation function.  
It is the *effective SNR available for delay estimation* given two noisy observations of the same signal.

### 4.3 The cross-correlation is the estimator, not the SNR source
The cross-correlation function is the *maximum likelihood estimator* (MLE) in many scenarios.  
It processes the separated signals to produce a delay estimate.

The CRB is a property of the **raw data likelihood** (the separated signals), not the estimator output.

The cross-correlation peak height and variance are *related* to this effective SNR, but the fundamental bound comes from the information content of the separated signals themselves.

### 4.4 Why this matters
If you think of SNR as the cross-correlation SNR, you are already in the "post-integration" regime (like \((WT/2\pi)\cdot\mathrm{SNR}\) in the Weinstein & Weiss papers).

That post-integration SNR determines the *threshold* behavior — whether the cross-correlator achieves the CRB or breaks down.

But the CRB itself is set by the **input SNR at each sensor**, combined through the effective SNR formula above.

**Summary:** The SNR is a property of the **separated received signals** at each sensor.  
The cross-correlation is an estimator that tries to extract the delay from those signals.  
The CRB is determined by the input SNR, not the output SNR of the correlator.

---

## 5. Important caveat
The CRB is tight only in the **ambiguity-free / high-SNR regime**.  
At lower SNR, threshold and ambiguity effects make the actual MSE much larger than the CRB.  
Ziv–Zakai and Weiss–Weinstein bounds are used to analyze that regime.

Source: Sadler & Kozick, “A Survey of Time Delay Estimation Performance Bounds,” Sections I–II.