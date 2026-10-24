# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 21:10:51 2025

@author: Seemab

"Enumeration of Tree-like Multigraphs with a Given Number of Vertices,
 Self-loops and Multiple Edges"
'Seemab Hayat'
Counts p(n, s, m, f<=, g<=, h<=, k<=) via DP, then applies
the unicentroid / bicentroid formulas.
"""

import numpy as np
import time

start_time = time.time()

# -----------------------------
# Parameter limits (edit here)
# -----------------------------
nv = 4 # max n
ns = 2 # max s
nm = 3# max m
nf = nv - 1
ng = ns
nh = nm
nk = nm



# -----------------------------
# DP tables
# -----------------------------
# p_eee[i,j,ℓ,f,g,h,k]  = p(i,j,ℓ, f=, g=, h=, k=)
# p_eel                = p(i,j,ℓ, f=, g=, h<=, k<=)
# p_ell                = p(i,j,ℓ, f=, g<=, h<=, k<=)
# p_lll                = p(i,j,ℓ, f<=, g<=, h<=, k<=)
p_eee = np.zeros((nv + 1, ns + 1, nm + 1,
                  nf + 1, ng + 1, nh + 1, nk + 1))
p_eel = np.zeros_like(p_eee)
p_ell = np.zeros_like(p_eee)
p_lll = np.zeros_like(p_eee)


def _ok(i2, j2, ell2, u2, v2, w2, x2):
    """Bounds checker for indices."""
    return (0 <= i2 <= nv and 0 <= j2 <= ns and 0 <= ell2 <= nm and
            0 <= u2 <= nf and 0 <= v2 <= ng and 0 <= w2 <= nh and
            0 <= x2 <= nk)


# =========================
# Base / initialization
# =========================

# ---- n = 1 base (corrected) ----
# For n = 1, there is exactly ONE rooted tree-like multigraph for each
# (s, m) with m = 0 and arbitrary s (all loops on the single vertex).
# Multiple edges between distinct vertices are impossible when n = 1,
# so states with ℓ > 0 stay 0.
for j in range(ns + 1):       # total self-loops s = j
    p_eee[1, j, 0, 0, 0, 0, 0] = 1.0

# ---- n = 2 base (Algorithm 1, line 3) ----
# For all 0 <= j <= ns, 0 <= ℓ <= nm, and 0 <= v <= j (g = v),
# we have one graph with:
#   n = 2, total self-loops = j, total multiple edges = ℓ,
#   Maxv = 1, Maxs = v, Maxm = 0, Maxℓ = ℓ   (x = ℓ)
#
# That is:
#   p[2, j, ℓ, 1=, v=, 0=, ℓ=] = 1
for j in range(ns + 1):       # total self-loops
    for ell in range(nm + 1): # total multiple edges
        for v in range(min(j, ng) + 1):
            # f=1, g=v, h=0, k=ell
            if 1 <= nf and 0 <= 0 <= nh and ell <= nk:
                p_eee[2, j, ell, 1, v, 0, ell] = 1.0


# -------------------------------------------------
# Helper: build ≤-versions for fixed n = i
#   p_eee[i,...]  ->  p_eel[i,...], p_ell[i,...], p_lll[i,...]
# -------------------------------------------------
def accumulate_for_n(i):
    # Start from exact states at this n
    p_eel[i] = p_eee[i]
    p_ell[i].fill(0.0)
    p_lll[i].fill(0.0)

    for j in range(ns + 1):
        for ell in range(nm + 1):

            # 1) Make h<=, k<= from exact h,k
            # First accumulate over h (w-index)
            for f in range(nf + 1):
                for g in range(ng + 1):
                    for h in range(1, nh + 1):
                        for k in range(nk + 1):
                            p_eel[i, j, ell, f, g, h, k] += \
                                p_eel[i, j, ell, f, g, h - 1, k]

            # Then accumulate over k (x-index)
            for f in range(nf + 1):
                for g in range(ng + 1):
                    for h in range(nh + 1):
                        for k in range(1, nk + 1):
                            p_eel[i, j, ell, f, g, h, k] += \
                                p_eel[i, j, ell, f, g, h, k - 1]

            # 2) Make g<= from p_eel to get p_ell
            for f in range(nf + 1):
                for g in range(ng + 1):
                    for h in range(nh + 1):
                        for k in range(nk + 1):
                            if g == 0:
                                p_ell[i, j, ell, f, 0, h, k] = \
                                    p_eel[i, j, ell, f, 0, h, k]
                            else:
                                p_ell[i, j, ell, f, g, h, k] = \
                                    p_ell[i, j, ell, f, g - 1, h, k] + \
                                    p_eel[i, j, ell, f, g, h, k]

            # 3) Make f<= from p_ell to get p_lll
            for f in range(nf + 1):
                for g in range(ng + 1):
                    for h in range(nh + 1):
                        for k in range(nk + 1):
                            if f == 0:
                                p_lll[i, j, ell, 0, g, h, k] = \
                                    p_ell[i, j, ell, 0, g, h, k]
                            else:
                                p_lll[i, j, ell, f, g, h, k] = \
                                    p_lll[i, j, ell, f - 1, g, h, k] + \
                                    p_ell[i, j, ell, f, g, h, k]


# Helper for states with w=, x<= (used in some cases)
def p_u_eq_v_eq_w_eq_x_le(i, j, ell, u, v, w, x_le):
    """
    Return p[i,j,ell, u=, v=, w=, x<=] using p_eel (which stores w<=, x<=).
    """
    if not (0 <= i <= nv and 0 <= j <= ns and 0 <= ell <= nm and
            0 <= u <= nf and 0 <= v <= ng and 0 <= w <= nh and
            0 <= x_le <= nk):
        return 0.0

    # For w = 0, 'w=' and 'w<=' coincide.
    if w == 0:
        return p_eel[i, j, ell, u, v, 0, x_le]

    # p_eel is cumulative in w, so subtract the (w-1) prefix.
    val = p_eel[i, j, ell, u, v, w, x_le]
    if w - 1 >= 0:
        val -= p_eel[i, j, ell, u, v, w - 1, x_le]
    return val


# Build ≤-versions for n = 1 and n = 2 (needed as subproblems)
accumulate_for_n(1)
accumulate_for_n(2)

# =========================
# Core DP for i ≥ 3
# =========================

for i in range(3, nv + 1):
    for j in range(ns + 1):           # total self-loops
        for ell in range(nm + 1):     # total multiple edges
            # u (Maxv/type size) must be <= i-1
            for u in range(1, min(i - 1, nf) + 1):          # Maxv
                for v in range(0, min(j, ng) + 1):          # Maxs (type)
                    for w in range(0, min(ell, nh) + 1):    # Maxm (type)
                        for x in range(0, min(ell, nk) + 1):  # Maxℓ (type)

                            # Feasibility conditions
                            # (g ≤ s and h + x ≤ m)
                            if v > j or w + x > ell:
                                continue

                            # If u = 1, can't have internal multiple edges (w≥1)
                            if u == 1 and w > 0:
                                continue

                            # Star/no-extra-structure case
                            if (j == 0 and ell == 0 and
                                v == 0 and w == 0 and x == 0 and u == 1):
                                p_eee[i, 0, 0, 1, 0, 0, 0] = 1.0
                                continue

                            # General case: components of type (u,v,w,x)
                            p_eee[i, j, ell, u, v, w, x] = 0.0

                            # Compute c (max y) from Lemma 1
                            bounds = [(i - 1) // u]          # vertex bound
                            if v > 0:
                                bounds.append(j // v)       # self-loop bound
                            if (w + x) > 0:
                                bounds.append(ell // (w + x))  # multiedge bound
                            c = min(bounds)
                            if c <= 0:
                                continue

                            # Number of rooted components of type (u,v,w,x)
                            f_idx = min(u - 1, nf)
                            g_idx = min(v, ng)
                            h_idx = min(w, nh)
                            k_idx = min(w, nk)
                            count_sub = p_lll[u, v, w, f_idx, g_idx, h_idx, k_idx]

                            d = 1.0
                            for q in range(1, c + 1):
                                # Multinomial factor
                                d = d * (count_sub + q - 1) / q

                                iu     = i   - q * u
                                jv     = j   - q * v
                                ell_w  = ell - q * w
                                ell_x  = ell - q * x
                                ell_wx = ell - q * (w + x)

                                if iu <= 0:
                                    break

                                # helper minima for f≤, g≤, h≤, k≤
                                u_min_iu    = min(iu - 1, u - 1)
                                v_min_jv    = min(jv, v - 1)     if v >= 1 else 0
                                w_min_ellw  = min(ell_w,  w - 1) if w >= 1 else 0
                                x_min_ellx  = min(ell_x,  x - 1) if x >= 1 else 0
                                w_min_ellwx = min(ell_wx, w - 1) if w >= 1 else 0
                                x_min_ellwx = min(ell_wx, x - 1) if x >= 1 else 0

                                # ---- 8 cases from Theorem 1 ----

                                # 1) v = 0, w = 0, x = 0
                                if v == 0 and w == 0 and x == 0:
                                    i2, j2, ell2 = iu, j, ell
                                    u2, v2, w2, x2 = u_min_iu, j, ell, ell
                                    if _ok(i2, j2, ell2, u2, v2, w2, x2):
                                        p_eee[i, j, ell, u, v, w, x] += \
                                            d * p_lll[i2, j2, ell2, u2, v2, w2, x2]

                                # 2) v >= 1, w = 0, x = 0
                                elif v >= 1 and w == 0 and x == 0:
                                    i2, j2, ell2 = iu, jv, ell
                                    # term with u=, v<=, w<=, x<=  -> p_ell
                                    u2, v2, w2, x2 = u, v_min_jv, ell, ell
                                    if _ok(i2, j2, ell2, u2, v2, w2, x2):
                                        p_eee[i, j, ell, u, v, w, x] += \
                                            d * p_ell[i2, j2, ell2, u2, v2, w2, x2]
                                    # term with u<=, v<=, w<=, x<= -> p_lll
                                    u2, v2, w2, x2 = u_min_iu, jv, ell, ell
                                    if _ok(i2, j2, ell2, u2, v2, w2, x2):
                                        p_eee[i, j, ell, u, v, w, x] += \
                                            d * p_lll[i2, j2, ell2, u2, v2, w2, x2]

                                # 3) v = 0, w >= 1, x = 0
                                elif v == 0 and w >= 1 and x == 0:
                                    i2, j2, ell2 = iu, j, ell_w
                                    # u=,v=,w<=,x<= -> p_eel
                                    u2, v2, w2, x2 = u, 0, w_min_ellw, ell_w
                                    if _ok(i2, j2, ell2, u2, v2, w2, x2):
                                        p_eee[i, j, ell, u, v, w, x] += \
                                            d * p_eel[i2, j2, ell2, u2, v2, w2, x2]
                                    # u<= -> p_lll
                                    u2, v2, w2, x2 = u_min_iu, j, ell_w, ell_w
                                    if _ok(i2, j2, ell2, u2, v2, w2, x2):
                                        p_eee[i, j, ell, u, v, w, x] += \
                                            d * p_lll[i2, j2, ell2, u2, v2, w2, x2]

                                # 4) v = 0, w = 0, x >= 1
                                elif v == 0 and w == 0 and x >= 1:
                                    i2, j2, ell2 = iu, j, ell_x
                                    # u=,v=,w<=,x<= (here w=0 so '=' and '<=' coincide)
                                    u2, v2, w2, x2 = u, 0, 0, x_min_ellx
                                    if _ok(i2, j2, ell2, u2, v2, w2, x2):
                                        p_eee[i, j, ell, u, v, w, x] += \
                                            d * p_eel[i2, j2, ell2, u2, v2, w2, x2]
                                    # u<= -> p_lll
                                    u2, v2, w2, x2 = u_min_iu, j, ell_x, ell_x
                                    if _ok(i2, j2, ell2, u2, v2, w2, x2):
                                        p_eee[i, j, ell, u, v, w, x] += \
                                            d * p_lll[i2, j2, ell2, u2, v2, w2, x2]

                                # 5) v >= 1, w >= 1, x = 0
                                elif v >= 1 and w >= 1 and x == 0:
                                    i2, j2, ell2 = iu, jv, ell_w
                                    # A: u=,v=,w<=,x<= -> p_eel
                                    u2, v2, w2, x2 = u, v, w_min_ellw, ell_w
                                    if _ok(i2, j2, ell2, u2, v2, w2, x2):
                                        p_eee[i, j, ell, u, v, w, x] += \
                                            d * p_eel[i2, j2, ell2, u2, v2, w2, x2]
                                    # B: u=,v<=,w<=,x<= -> p_ell
                                    u2, v2, w2, x2 = u, v_min_jv, ell_w, ell_w
                                    if _ok(i2, j2, ell2, u2, v2, w2, x2):
                                        p_eee[i, j, ell, u, v, w, x] += \
                                            d * p_ell[i2, j2, ell2, u2, v2, w2, x2]
                                    # C: u<= -> p_lll
                                    u2, v2, w2, x2 = u_min_iu, jv, ell_w, ell_w
                                    if _ok(i2, j2, ell2, u2, v2, w2, x2):
                                        p_eee[i, j, ell, u, v, w, x] += \
                                            d * p_lll[i2, j2, ell2, u2, v2, w2, x2]

                                # 6) v >= 1, w = 0, x >= 1
                                elif v >= 1 and w == 0 and x >= 1:
                                    i2, j2, ell2 = iu, jv, ell_x
                                    # A: u=,v=,w<=,x<= (w=0)
                                    u2, v2, w2, x2 = u, v, 0, x_min_ellx
                                    if _ok(i2, j2, ell2, u2, v2, w2, x2):
                                        p_eee[i, j, ell, u, v, w, x] += \
                                            d * p_eel[i2, j2, ell2, u2, v2, w2, x2]
                                    # B: u=,v<= -> p_ell
                                    u2, v2, w2, x2 = u, v_min_jv, ell_x, ell_x
                                    if _ok(i2, j2, ell2, u2, v2, w2, x2):
                                        p_eee[i, j, ell, u, v, w, x] += \
                                            d * p_ell[i2, j2, ell2, u2, v2, w2, x2]
                                    # C: u<= -> p_lll
                                    u2, v2, w2, x2 = u_min_iu, jv, ell_x, ell_x
                                    if _ok(i2, j2, ell2, u2, v2, w2, x2):
                                        p_eee[i, j, ell, u, v, w, x] += \
                                            d * p_lll[i2, j2, ell2, u2, v2, w2, x2]

                                # 7) v = 0, w >= 1, x >= 1
                                elif v == 0 and w >= 1 and x >= 1:
                                    i2, j2, ell2 = iu, j, ell_wx
                                    # A: u=,v=,w=,x<=  -> reconstruct from p_eel
                                    u2, v2, w2, x2 = u, 0, w, x_min_ellwx
                                    tmp = p_u_eq_v_eq_w_eq_x_le(i2, j2, ell2,
                                                                u2, v2, w2, x2)
                                    p_eee[i, j, ell, u, v, w, x] += d * tmp
                                    # B: u=,v=,w<=,x<= -> p_eel
                                    u2, v2, w2, x2 = u, 0, w_min_ellwx, ell_wx
                                    if _ok(i2, j2, ell2, u2, v2, w2, x2):
                                        p_eee[i, j, ell, u, v, w, x] += \
                                            d * p_eel[i2, j2, ell2, u2, v2, w2, x2]
                                    # C: u<= -> p_lll
                                    u2, v2, w2, x2 = u_min_iu, j, ell_wx, ell_wx
                                    if _ok(i2, j2, ell2, u2, v2, w2, x2):
                                        p_eee[i, j, ell, u, v, w, x] += \
                                            d * p_lll[i2, j2, ell2, u2, v2, w2, x2]

                                # 8) v >= 1, w >= 1, x >= 1
                                elif v >= 1 and w >= 1 and x >= 1:
                                    i2, j2, ell2 = iu, jv, ell_wx
                                    # A: u=,v=,w=,x<=
                                    u2, v2, w2, x2 = u, v, w, x_min_ellwx
                                    tmp = p_u_eq_v_eq_w_eq_x_le(i2, j2, ell2,
                                                                u2, v2, w2, x2)
                                    p_eee[i, j, ell, u, v, w, x] += d * tmp
                                    # B: u=,v=,w<=,x<= -> p_eel
                                    u2, v2, w2, x2 = u, v, w_min_ellwx, ell_wx
                                    if _ok(i2, j2, ell2, u2, v2, w2, x2):
                                        p_eee[i, j, ell, u, v, w, x] += \
                                            d * p_eel[i2, j2, ell2, u2, v2, w2, x2]
                                    # C: u=,v<= -> p_ell
                                    u2, v2, w2, x2 = u, v_min_jv, ell_wx, ell_wx
                                    if _ok(i2, j2, ell2, u2, v2, w2, x2):
                                        p_eee[i, j, ell, u, v, w, x] += \
                                            d * p_ell[i2, j2, ell2, u2, v2, w2, x2]
                                    # D: u<= -> p_lll
                                    u2, v2, w2, x2 = u_min_iu, j, ell_wx, ell_wx
                                    if _ok(i2, j2, ell2, u2, v2, w2, x2):
                                        p_eee[i, j, ell, u, v, w, x] += \
                                            d * p_lll[i2, j2, ell2, u2, v2, w2, x2]

    # After finishing p_eee[i,...], build ≤-versions for this i
    accumulate_for_n(i)

print(f"DP finished for nv = {nv}")
# ============================================================
# Helper wrapper for p(n,s,m,f<=,g<=,h<=,k<=)
# ============================================================

def p_query(n, s, m, f_le, g_le, h_le, k_le):
    """
    Safe wrapper for p(n,s,m,f<=,g<=,h<=,k<=) using p_lll.
    Clips all indices to the table bounds.
    """
    if n < 0 or s < 0 or m < 0:
        return 0
    if n > nv or s > ns or m > nm:
        return 0

    f_le = min(f_le, nf)
    g_le = min(g_le, ng)
    h_le = min(h_le, nh)
    k_le = min(k_le, nk)

    return p_lll[n, s, m, f_le, g_le, h_le, k_le]

def count_trees(n, s, m):
    """
    Combined counting function with symmetric pairing of halves.
    """

    # --- Unicentroid ---
    f_le = (n - 1) // 2
    unicentroid = p_query(n, s, m, f_le, s, m, m)

    # odd n ⇒ no bicentroid
    if n % 2 == 1:
        return unicentroid, 0, unicentroid

    # --- Bicentroid ---
    half = n // 2
    half_f_le = half - 1

    def P_half(s_sub, m_sub):
        return p_query(half, s_sub, m_sub,
                       half_f_le, s_sub, m_sub, m_sub)

    bicentroid = 0

    # symmetric pairing over (s_x, m_x, ell)
    for s_x in range(0, s + 1):
        for m_x in range(0, m + 1):

            for ell_used in range(0, m - m_x + 1):
                s_y = s - s_x
                m_y = m - m_x - ell_used
                if s_y < 0 or m_y < 0:
                    continue

                X = (s_x, m_x)
                Y = (s_y, m_y)

                # unordered pair condition
                if X > Y:
                    continue

                P_x = P_half(s_x, m_x)
                if P_x == 0:
                    continue
                P_y = P_half(s_y, m_y)
                if P_y == 0:
                    continue

                if X < Y:
                    bicentroid += P_x * P_y
                else:
                    bicentroid += (P_x * (P_x + 1)) // 2

    total = unicentroid + bicentroid
    return unicentroid, bicentroid, total
uni_case, bic_case, total_number = count_trees(nv, ns, nm)

print("\n================ FINAL RESULTS ================")
print(f"Input (n, s, m) = ({nv}, {ns}, {nm})")
print("----------------------------------------------")
print(f"Unicentroid contribution : {uni_case}")
print(f"Bicentroid contribution  : {bic_case}")
print(f"Total number of trees    : {total_number}")
print("===============================================\n")

end_time = time.time()
print(f"Total computation time: {end_time - start_time:.4f} seconds\n")

