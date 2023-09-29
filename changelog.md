# `elk` Changelog
This log keeps track of the changes implemented in each version of `elk`.

## 1.0
- Fixing typos in doccumentation, and acceptence of the JOSS paper.
- Creating compatability with M1 processors

## 0.10 
- Changing the required version of Numpy to <1.25 which was incompatible with the version of lightkurve we support. 
- Change the X axis for Auto-Correlation Function to be sequential. 

## 0.9
- Fixing bug in the from_fits functions, making new tutorial and minor updates to other tutorials

## 0.8
- Updating the summary table output to handle different length of columns in different clusters, and fix bug issues with 0.7

## 0.7 
- Updating the summary table output to handle different length of columns in different clusters 

## 0.6 
- Updating the way the new summary table handles cases where there were no good sectors

## 0.5
- There is a '/' at the start of the path for `from_fits`

## 0.4
- Set the output path for classes read in using `from_fits`

## 0.3
- Separate `lightcurves_summary_file` into separate functions to avoid recreating things

## 0.2
- Automate testing and add JOSS compilation script by @TomWagg in https://github.com/tobin-wainer/elk/pull/92
- Switch condition order by @tobin-wainer in https://github.com/tobin-wainer/elk/pull/93

## 0.1
- Fixed error in printing tables. 
- Fixed Simbad Query return See #90 

## 0.0
- Initial release
