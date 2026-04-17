"""
Closed-form 2D scattering-width series for circular cylinders.

All formulas use the e^{-jwt} time convention to match the solver.
Results are returned as sigma_2D (meters), the 2D scattering width per unit
length, which is what the solver returns as "rcs_linear".

Conventions:
- TE = E_z polarization (electric field along cylinder axis) = Dirichlet-like
  on PEC.  This matches the solver's "TE" / "VV" label.
- TM = H_z polarization (magnetic field along cylinder axis) = Neumann-like
  on PEC.  This matches the solver's "TM" / "HH" label.

References:
- Balanis, "Advanced Engineering Electromagnetics," 2nd ed., Ch. 11.
- Harrington, "Time-Harmonic Electromagnetic Fields," Ch. 5.
- Bohren & Huffman, "Absorption and Scattering of Light by Small Particles,"
  Ch. 8 (infinite-cylinder normal incidence).

All series return monostatic backscatter sigma_2D in meters.
"""

import numpy as np
from scipy import special as sp


C0 = 299_792_458.0


def _hankel2(n, z):
    """Hankel function of the second kind, order n, argument z."""
    return sp.hankel2(n, z)


def _jn(n, z):
    return sp.jn(n, z)


def _jn_prime(n, z):
    """Derivative of Bessel J_n w.r.t. argument.  Array-safe."""
    return sp.jvp(n, z, 1)


def _hankel2_prime(n, z):
    """Derivative of Hankel H_n^{(2)} w.r.t. argument.  Array-safe."""
    return sp.h2vp(n, z, 1)


def _nmax_for_ka(ka, pad=10):
    """Safe series truncation. Wiscombe's rule + pad."""
    ka_abs = abs(complex(ka))
    n = int(np.ceil(ka_abs + 4.05 * ka_abs**(1.0 / 3.0) + 2.0))
    return max(10, n + pad)


def sigma_pec_cylinder(radius_m, freq_hz, polarization):
    """
    Monostatic 2D backscatter sigma_2D for a PEC circular cylinder.

    Parameters
    ----------
    radius_m : float
        Cylinder radius in meters.
    freq_hz : float
        Frequency in Hz.
    polarization : str
        'TE' (E_z axial, Dirichlet) or 'TM' (H_z axial, Neumann).

    Returns
    -------
    sigma_2D in meters.
    """
    pol = polarization.upper()
    k = 2.0 * np.pi * freq_hz / C0
    ka = k * radius_m
    N = _nmax_for_ka(ka)

    if pol == 'TE':
        # E_z polarization.  BC: E_z = 0 on PEC, i.e., total field vanishes.
        # Scattered-field expansion coefficient:
        #   a_n = -J_n(ka) / H_n^{(2)}(ka)
        # Monostatic scattered field amplitude at phi = pi (backscatter):
        #   f(phi=pi) = sum_{n=-inf}^{inf} a_n e^{jn pi}
        n_arr = np.arange(-N, N + 1)
        a_n = -_jn(n_arr, ka) / _hankel2(n_arr, ka)
        # Backscatter factor: e^{jn pi} = (-1)^n
        amp = np.sum(a_n * (-1.0)**n_arr)
    elif pol == 'TM':
        # H_z polarization.  BC: dH_z/drho = 0 on PEC.
        #   a_n = -J_n'(ka) / H_n^{(2)}'(ka)
        n_arr = np.arange(-N, N + 1)
        a_n = -_jn_prime(n_arr, ka) / _hankel2_prime(n_arr, ka)
        amp = np.sum(a_n * (-1.0)**n_arr)
    else:
        raise ValueError(f"Unknown polarization {polarization}")

    # 2D scattering width (RCS per unit length):
    #   sigma_2D = (4 / k) * |sum a_n e^{j n phi}|^2
    sigma = (4.0 / k) * abs(amp)**2
    return float(sigma)


def sigma_dielectric_cylinder(radius_m, eps_r, mu_r, freq_hz, polarization):
    """
    Monostatic 2D backscatter sigma_2D for a homogeneous dielectric cylinder
    in free space.

    Uses the standard infinite-cylinder series (Bohren & Huffman Ch. 8,
    normal incidence) adapted to 2D scattering width and e^{-jwt} convention.

    Parameters
    ----------
    radius_m : float
        Cylinder radius in meters.
    eps_r, mu_r : complex
        Relative permittivity / permeability.  For lossy media, use
        eps_r with NEGATIVE imaginary part to match e^{-jwt}.
    freq_hz : float
        Frequency in Hz.
    polarization : str
        'TE' (E_z axial) or 'TM' (H_z axial).
    """
    pol = polarization.upper()
    k0 = 2.0 * np.pi * freq_hz / C0
    n_rel = np.sqrt(complex(eps_r) * complex(mu_r))
    k1 = k0 * n_rel  # wavenumber inside dielectric
    x = k0 * radius_m
    mx = k1 * radius_m
    N = _nmax_for_ka(max(abs(x), abs(mx)))

    # Match traces of the scalar field u (E_z for TE, H_z for TM) and
    # a polarization-dependent normal-derivative operator across rho = a.
    #
    # TE (E_z): continuity of E_z and (1/mu) dE_z/drho
    # TM (H_z): continuity of H_z and (1/eps) dH_z/drho
    #
    # Inside:     u_n^in  = c_n J_n(k1 rho)
    # Outside:    u_n^out = J_n(k0 rho) + a_n H_n^(2)(k0 rho)
    #
    # Matching at rho = a gives a 2x2 system for (a_n, c_n).
    #
    # Writing the "transverse impedance" factor
    #   TE: zeta_i = mu_i, so RHS uses (k_i / mu_i)
    #   TM: zeta_i = eps_i, so RHS uses (k_i / eps_i)

    if pol == 'TE':
        zeta_out = 1.0  # mu0 = 1 (relative)
        zeta_in = complex(mu_r)
    elif pol == 'TM':
        zeta_out = 1.0  # eps0 = 1 (relative)
        zeta_in = complex(eps_r)
    else:
        raise ValueError(f"Unknown polarization {polarization}")

    n_arr = np.arange(-N, N + 1)
    # Scattered-field coefficients a_n from the boundary match.
    # System:
    #   J_n(x) + a_n H_n(x)           = c_n J_n(mx)
    #   (k0/zeta_out) [J_n'(x) + a_n H_n'(x)] = (k1/zeta_in) c_n J_n'(mx)
    # Solve for a_n:
    #   a_n = -[ (k1/zeta_in) J_n'(mx) J_n(x) - (k0/zeta_out) J_n(mx) J_n'(x) ]
    #         / [ (k1/zeta_in) J_n'(mx) H_n(x) - (k0/zeta_out) J_n(mx) H_n'(x) ]
    Jn_x  = _jn(n_arr, x)
    Jnp_x = np.array([_jn_prime(int(n), x) for n in n_arr])
    Hn_x  = _hankel2(n_arr, x)
    Hnp_x = np.array([_hankel2_prime(int(n), x) for n in n_arr])
    Jn_mx = _jn(n_arr, mx)
    Jnp_mx = np.array([_jn_prime(int(n), mx) for n in n_arr])

    p = k1 / zeta_in
    q = k0 / zeta_out

    num = -(p * Jnp_mx * Jn_x - q * Jn_mx * Jnp_x)
    den = (p * Jnp_mx * Hn_x - q * Jn_mx * Hnp_x)
    a_n = num / den

    amp = np.sum(a_n * (-1.0)**n_arr)
    sigma = (4.0 / k0) * abs(amp)**2
    return float(sigma)


def sigma_coated_pec_cylinder(a_inner_m, a_outer_m, eps_r, mu_r,
                              freq_hz, polarization):
    """
    Monostatic 2D backscatter sigma_2D for a PEC cylinder of radius
    a_inner coated with a homogeneous dielectric out to a_outer.

    Three-medium problem:
      region 0 (rho > a_outer): air (eps=mu=1)
      region 1 (a_inner < rho < a_outer): dielectric (eps_r, mu_r)
      region 2 (rho < a_inner): PEC

    Field expansions per mode n:
      Outside:    J_n(k0 rho) + a_n H_n^(2)(k0 rho)
      Coating:    b_n J_n(k1 rho) + d_n Y_n(k1 rho)
                  [or equivalently H_n^(1) and H_n^(2); we use Jn, Yn]
      Inside PEC: vanishes (boundary condition)

    Boundary conditions:
      at rho = a_inner:
         TE: coating field = 0        [E_z = 0 on PEC]
         TM: d(coating)/drho = 0      [dH_z/drho = 0 on PEC]
      at rho = a_outer:
         coating trace = outside trace
         (k/zeta)*coating_normal_deriv = (k/zeta)*outside_normal_deriv
    """
    pol = polarization.upper()
    k0 = 2.0 * np.pi * freq_hz / C0
    n_rel = np.sqrt(complex(eps_r) * complex(mu_r))
    k1 = k0 * n_rel

    x_in = k1 * a_inner_m       # argument at PEC surface (inside coating)
    x_out_in = k1 * a_outer_m   # argument at outer surface (inside coating)
    x_out = k0 * a_outer_m      # argument at outer surface (outside / air)

    N = _nmax_for_ka(max(abs(x_in), abs(x_out_in), abs(x_out)))

    if pol == 'TE':
        zeta_out = 1.0; zeta_in = complex(mu_r)
    elif pol == 'TM':
        zeta_out = 1.0; zeta_in = complex(eps_r)
    else:
        raise ValueError(polarization)

    p = k1 / zeta_in
    q = k0 / zeta_out

    n_arr = np.arange(-N, N + 1)
    a_n = np.zeros(n_arr.shape, dtype=complex)

    for idx, n in enumerate(n_arr):
        n = int(n)
        # Coating basis: J_n(k1 rho) and Y_n(k1 rho).
        # Use sp.jv / sp.yv which accept complex arguments.
        Jn_in   = sp.jv(n, x_in)
        Yn_in   = sp.yv(n, x_in)
        Jnp_in  = sp.jvp(n, x_in, 1)
        Ynp_in  = sp.yvp(n, x_in, 1)
        Jn_oi   = sp.jv(n, x_out_in)
        Yn_oi   = sp.yv(n, x_out_in)
        Jnp_oi  = sp.jvp(n, x_out_in, 1)
        Ynp_oi  = sp.yvp(n, x_out_in, 1)
        Jn_o    = sp.jv(n, x_out)
        Jnp_o   = sp.jvp(n, x_out, 1)
        Hn_o    = sp.hankel2(n, x_out)
        Hnp_o   = sp.h2vp(n, x_out, 1)

        # PEC BC at rho = a_inner.
        # TE:  b Jn_in + d Yn_in = 0  =>  d = -b Jn_in/Yn_in
        # TM:  b Jnp_in + d Ynp_in = 0 => d = -b Jnp_in/Ynp_in
        if pol == 'TE':
            alpha = -Jn_in / Yn_in
        else:
            alpha = -Jnp_in / Ynp_in

        # Combined coating basis: phi_n(rho) = J_n(k1 rho) + alpha Y_n(k1 rho).
        # Its value and radial derivative at rho = a_outer:
        phi_o   = Jn_oi  + alpha * Yn_oi
        phip_o  = Jnp_oi + alpha * Ynp_oi

        # Outer matching: two equations in (a_n, b_n):
        #   Jn_o + a_n Hn_o = b_n phi_o
        #   q*(Jnp_o + a_n Hnp_o) = p * b_n * phip_o
        # Solve for a_n:
        num = -(p * phip_o * Jn_o - q * phi_o * Jnp_o)
        den = (p * phip_o * Hn_o - q * phi_o * Hnp_o)
        a_n[idx] = num / den

    amp = np.sum(a_n * (-1.0)**n_arr)
    sigma = (4.0 / k0) * abs(amp)**2
    return float(sigma)
