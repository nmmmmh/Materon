# -*- coding: utf-8 -*-
"""Schema utilities for the TF2.9 multi-backbone engine."""

RAW_FEATURE_COLUMNS = ['density', 'atom_melting_point', 'atom_specific_heat_capacity', 'atomic_boiling_point', 'atomic_fusion_heat', 'atomic_thermal_cond', 'atomic_ip', 'atomic_ea', 'atomic_hfomo', 'atomic_lfumo', 'X', 'atomic_pol', 'atomic_nvalences', 'atomic_ndval', 'atomic_npval', 'atomic_nsval', 'atomic_radius', 'atomic_covalent Radius', 'atomic_volume', 'atomic_weight']
FEATURE_GROUPS = {'mass': ['atomic_weight'], 'bulk_density': ['density'], 'size': ['atomic_radius', 'atomic_covalent Radius', 'atomic_volume'], 'thermal_phase_energy': ['atom_melting_point', 'atomic_boiling_point', 'atomic_fusion_heat'], 'heat_capacity': ['atom_specific_heat_capacity'], 'thermal_transport': ['atomic_thermal_cond'], 'ionization_affinity': ['atomic_ip', 'atomic_ea'], 'frontier_orbitals': ['atomic_hfomo', 'atomic_lfumo'], 'electronegativity': ['X'], 'polarizability': ['atomic_pol'], 'valence_total': ['atomic_nvalences'], 'valence_orbitals': ['atomic_nsval', 'atomic_npval', 'atomic_ndval']}
MAIN_CONTEXT_MODE = "grouped_pca"
