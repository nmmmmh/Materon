# MoTaTe₂/MoS₂ interface: analysis plan fixed before results (2026-11-19)

Written before any interface result is read, so that the manuscript wording cannot be tuned to the outcome. Record the SHA-256 of this file next to the job directory when the first static calculation finishes.

## Systems
- MoTaTe₂/MoS₂ and Ta₂Te₂/MoS₂ in the same lateral supercell, PBE + D3, dipole correction, the k-point density and cutoff of the paper.
- Report the in-plane strain applied to each layer for both systems.
- All stacking registries that are computed are reported. The lowest-energy registry is the reference; the spread over registries is quoted.

## Quantities (defined now)
1. n-type Schottky barrier Φ_B,n = E_CBM(MoS₂ layer) − E_F, with E_CBM read from the MoS₂-projected DOS of the heterostructure (onset at a fixed threshold, the same for both metals).
2. Interface dipole ΔV, the step of the plane-averaged electrostatic potential across the interface, relative to the sum of the isolated, strained layers.
3. Charge transfer from the plane-averaged density difference, integrated up to the plane of the minimum.
4. Pinning slope S = ΔΦ_B,n / ΔΦ_M between the two metals, using the work functions of the isolated strained layers.
5. The tunnel-barrier height and width at the van der Waals gap, from the same potential.

## Decision rule for the manuscript
- If Φ_B,n(MoTaTe₂) ≤ 0, state that the contact remains ohmic at the interface, and drop the "next step" sentence from the conclusion.
- If Φ_B,n(MoTaTe₂) > 0, state the interface barrier for both metals, the barrier difference and S, in that order. Do not present the result as a "reduction" without giving S.
- In both cases the comparison between the two metals in the same cell is the headline, because errors in the functional largely cancel in that difference.
- Figure 4(c,d) become the layer-resolved DOS and plane-averaged potential of the interface. The current single-layer level diagram moves to the SI.
