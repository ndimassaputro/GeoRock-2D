# GeoRock-2D

## Uncertainty-Aware Seismic Rockhead Characterization Using Synthetic Travel-Time Data and Sparse Boreholes

GeoRock-2D is an independent computational geophysics mini-research project developed to investigate how synthetic seismic travel-time information and sparse borehole constraints affect the accuracy and uncertainty of two-dimensional engineering rockhead characterization.

The project integrates synthetic geological modelling, travel-time forward modelling, regularized slowness inversion, ray-coverage analysis, engineering rockhead extraction, sparse-borehole constraints, and empirical Monte Carlo uncertainty analysis in a reproducible Python workflow.

All geological models and observations used in this repository are synthetic. The project does not propose a new inversion algorithm and has not been validated using field data.

---

## Research Question

How does the integration of seismic travel-time information and sparse borehole constraints affect the accuracy and uncertainty of 2D engineering rockhead characterization?

---

## Scientific Motivation

Borehole investigations provide direct subsurface observations but are spatially sparse. Geophysical measurements can provide more continuous information between boreholes, although their interpretation remains affected by non-uniqueness, measurement noise, survey geometry, and regularization.

This project evaluates the complementary roles of:

- seismic travel-time information;
- ray-path coverage;
- regularized inversion;
- sparse borehole observations;
- uncertainty quantification.

---

## Workflow

```text
Synthetic 2D velocity model
            ↓
Surface acquisition geometry
            ↓
Conceptual travel-time forward model
            ↓
Sensitivity matrix G
            ↓
Noisy synthetic observations
            ↓
Regularized slowness inversion
            ↓
Coverage-aware rockhead extraction
            ↓
Sparse borehole constraints
            ↓
Monte Carlo uncertainty analysis
            ↓
Engineering interpretation