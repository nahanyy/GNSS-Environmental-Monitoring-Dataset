GNSS Environmental Monitoring Dataset Processing Code
1. Overview

This repository provides the preprocessing and visualization codes for the GNSS Environmental Monitoring Dataset.

The released dataset is generated from raw u-blox GNSS receiver outputs through a two-stage processing pipeline:

UBX message decoding and organization

Raw receiver messages are converted into structured observation files, satellite information files, and positioning solution files.

Satellite-position-based alignment and feature matrix generation

The processed GNSS observations are aligned according to satellite elevation and azimuth angles and transformed into satellite-time matrices for subsequent GNSS environmental sensing and signal analysis.

The processing workflow preserves the original GNSS observation characteristics, including multipath, blockage, and NLOS-related effects, without applying observation-level quality filtering.
