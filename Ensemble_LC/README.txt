This is the folder dedicated to the pipeline I created to download and correct ensemble light curves of star clusters, and calculate their variabilty metrics.

A detailed account and example of each pipeline can be found in the Jupyter Notebooks titled "Detailed and Simple Pipeline to get Light curves", and "Detailed and Simple Pipeline to Calculate Variability Statistics from Light Curves" respectively.

For the Light curve Pipeline:
All of the required functions are combined in the lightcurves_pipeline.py file 

*THE ONLY FUNCTION TO INPORT AND USE IS "Get_Corrected_Lightcurve"* 
This can be accoplished by using the following code: 'from Ensemble_LC.lightcurves_pipeline import Get_Corrected_Lightcurve'

*THIS CODE MUST BE INITIALIZED BY SETTING A "Path_to_Save_to" in the lightcurves_pipeline.py file*

For the Variabilty state pipeline:
All of the required functions are in the variable_stats.py file. Each of these funcitons can be used seperately, or a comprehensive table can be made using the "Get_Variable_Stats_Table" function.  