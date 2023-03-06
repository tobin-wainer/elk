# ELK
This Repo is for the Python package ELK, making integrated light, lightcurves from TESS Data.

This package includes the pipelines I created to download and correct ensemble light curves of star clusters in TESS data. While I have used ELK for star cluster science, it can be used by any user to create ensemble light curves for any given aperature. 


## TODO: Update all of this

A detailed account and example of each pipeline can be found in the Jupyter Notebooks titled "Detailed and Simple Pipeline to get Light curves", "Detailed and Simple Pipeline to Generate All Sector Lightcurves", and "Detailed and Simple Pipeline to Calculate Variability Statistics from Light Curves" respectively.

The pipeline which I recommend using is the one to generate all sector lightcurves. 

**THE FUNCTION TO INPORT AND USE IS "Generate_Lightcurves"** 

**THIS CODE MUST BE INITIALIZED BY SETTING A "Path_to_Save_to" in the lightcurves_pipeline.py file**

There is also a function called "Access Lightcurve" which can be used to pull the lightcurve table, and the figure of the lightcurve.


For the pipeline to get any sector Light curve:
All of the required functions are combined in the one_lightcurve_pipeline.py file 

**THE ONLY FUNCTION TO INPORT AND USE IS "Get_Corrected_Lightcurve"** 
This can be accoplished by using the following code: 'from Ensemble_LC.lightcurves_pipeline import Get_Corrected_Lightcurve'

**THIS CODE MUST BE INITIALIZED BY SETTING A "Path_to_Save_to" in the one_lightcurve_pipeline.py file**

For the Variabilty state pipeline:
All of the required functions are in the variable_stats.py file. Each of these funcitons can be used seperately, or a comprehensive table can be made using the "Get_Variable_Stats_Table" function.  

**THIS CODE MUST BE INITIALIZED BY SETTING A "Path_to_Save_to" AND A "Path_to_Read_in_LCs" in the variable_stats.py file**
