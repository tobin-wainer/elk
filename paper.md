---
title: 'ELK: A python package for correcting, analyzing, and diagnosing TESS integrated light curves'
tags:
  - Python
  - astronomy
  - variabilty
  - blending
  - star clusters
authors:
  - name: Tobin M. Wainer
    orcid: 0000-0001-6320-2230
    equal-contrib: true
    affiliation: "1, 2"
  - name: Tom Wagg
    orcid: 0000-0001-6147-5761
    affiliation: "1"
  - name: Vijith Jacob Poovelil
    orcid: 0000-0002-9831-3501
    affiliation: "2"
  - name: Gail Zasowski
    orcid: 0000-0001-6761-9359
    affiliation: "2"
affiliations:
 - name: Department of Astronomy, University of Washington, Box 351580, Seattle, WA 98195, USA
   index: 1
 - name: Department of Physics and Astronomy, University of Utah, Salt Lake City, UT 84112, USA
   index: 2
date: 07 June 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal
---

# Summary

`elk` (int**E**grated **L**ight **K**urve) is an open-source Python package designed for downloading and correcting *integrated* light curves from the Full Frame Images (FFI) of TESS data. We provide novel approaches for the analysis of heavily blended light curves. Our correction is in the form of aperture photometry, building upon the methodology of the Python package `lightkurve` [@lightkurve_collaboration_lightkurve_2018]. 

The term `integrated' refers to the method in which we are extracting blended photometry from an aperture, a practice common in the literature. Our approach does this extraction at many time steps to generate light curves, and we evaluate how the integrated light changes through time. Each pixel light curve is determined independently, and then summed to create the integrated light curve.

Our correction techniques utilizes the bona fide pipeline `lightkurve`, and are able to account for spatially-transient scattered light from the Earth and Moon (using singular value decomposition methods and principle component analysis), known TESS telescope systems (encoding in co-trending basis vectors) and long-period astrophysical variability (through the application of a basis-spline model). Each of these correction techniques from the `lightkurve` package are flexible within `elk` and can be tuned or turned off entirely. `elk` easily wraps the multiple components from `lightkurve` into a single, easy to use package for integrated light curves, a use case not common within existing extraction pipelines like `lighkurve` or `eleanor` [@feinstein_eleanor_2019].

`elk` additionally has a range of features to analyze integrated light curves. We include a wide variety of variability metrics from across astronomical subfields, including the Stetson J statistic and von Neumann ratio [@stetson_center_1994; @von_neumann_distribution_1941]. `elk` features simple and useful diagnostic capabilities, with flexible functions for the creation and visualisation of bootstrapped Lomb-Scargle periodograms and autocorrelation functions. Moreover, one can use `elk` to locate the precise sky position of a variable signature causing a feature in an integrated light curve, and identify likely candidate stars from SIMBAD [@wenger_simbad_2000]. Though we designed `elk` with the intention of studying cluster variability, it has the flexibility to be applied to *any* integrated study.

Through `elk`, users have a single package with which to create, analyze and diagnose reliable integrated light curves. 

# Statement of need

One important property of stars is their intrinsic photometric variability, including both high- and low-amplitude variations driven by coherent pulsations and other physical mechanisms like rotation. A great deal of the foundational work on stellar variability, especially of the low-amplitude variety, has been done in the Milky Way, including the recent revolution in asteroseismic measurements of stars across the Galaxy (e.g., @chaplin_asteroseismic_2013; @pinsonneault_second_2018). High-amplitude pulsational variables such as Cepheids and RR~Lyrae associated with clusters have long been used to measure distances and provide constraints on foreground extinction (e.g., @alonso-garcia_variable_2021). Famously, it was Cepheid variables that were used to measure the distance to M31 and to argue that the then-called "island universes" and "spiral nebulae" were in fact other galaxies [@hubble_spiral_1929]. Because of the high impact uses of stellar variability, a number of globular clusters in the Milky Way have detailed variable star membership catalogs (e.g., @clement_variable_2001). 

For environments where individual stars can not be resolved, the current standard approach to population analysis is through integrated light methods. However, the variability of unresolved stellar populations, i.e., of their integrated light, remain extremely limited. One analysis of unresolved field populations in M87 [@conroy_ubiquitous_2015] characterized the density of long-period variable stars, as a prediction for the age of the population. The main issue with using integrated light for variability encountered in this study was the blending of sources. There have been some efforts to address this issue (in TESS, relevant for this current work) for the light curves of individual resolved stars (e.g., @oelkers_precision_2018; @nardiello_psf-based_2020; @higgins_localizing_2022). The work presented here in `elk`, sidesteps these issues by working directly with the integrated light curves.

While the primary mission of TESS was the discovery of exoplanets, the unrivaled precision of its time series photometry led to many significant advances in stellar science. For example, TESS has been influential in the recent surge of new astroseismology results (e.g., @handberg_tess_2021). The 200,000 primary targets in the TESS input catalog (TIC, @stassun_tess_2018) were selected to detect exoplanets, and are thus not heavily blended. However, due to the large TESS pixels the FFIs contain tens of millions of sources, of which many are blended. These blended sources still contain bountiful astrophysical information and there have been many studies focused on blended star extraction (e.g., @oelkers_precision_2018; @nardiello_psf-based_2020; @higgins_localizing_2022). For the FFIs, more than 20 million sources for which relative photometry with $1\%$ photometric precision can be obtained [@ricker_transiting_2015, @huang_photometry_2020, @kunimoto_quick-look_2021]. These data provide a rich data set which have been used across astronomy subfields, and thus, the treatment of these data have been the topic of recent studies [@hattori_unpopular_2022].

`elk` is a python package designed to harness the bountiful information available in blended FFI photometry in the form of aperture based light curves. This package is the first of its kind for the correction, analysis and diagnosis of *integrated* light curves.

While most studies in the literature seek to extract individual sources from blended photometry, `elk` bypasses the problems associated with this type of analysis and instead analyses the light curves on the whole. Given this structure, once a light curve is corrected and identified to have a variability signature, it becomes increasingly important to know where that signature is coming from within the aperture. `elk` uses novel methodology to identify which pixels in the aperture are contributing to specific peaks in the LSP, and will then return a SIMBAD query of the stars contained within that TESS pixel. This allows users to understand the impact specific stars have in their integrated light curves.

The open source nature of this code allows users of all skill levels to generate precise light curves from aperture based sources, with minimal input. This gives users the ability to fully diagnose the light curves in one convenient location. 

`elk` is jointly published in [@wainer_catalog_2023], and further scientific uses of the package are explored in this work. 

# Acknowledgements

We would like to recognize and thank Samantha-Lynn Martinez for designing the `elk` logo.
TMW, GZ, and VJP acknowledge support by the Heising-Simons Foundation through grant 2019-1492. 
This paper includes data collected with the TESS mission, obtained from the MAST data archive at the Space Telescope Science Institute (STScI). Funding for the TESS mission is provided by the NASA's Science Mission Directorate. STScI is operated by the Association of Universities for Research in Astronomy, Inc., under NASA contract NAS 5–26555.

# References
