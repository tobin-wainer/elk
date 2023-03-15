import astropy.units as u
from elk.ensemble import EnsembleLC

c = EnsembleLC(output_path="../../output",
               cluster_name='NGC 419',
               location='23.58271, +61.1236',
               radius=.046,
               cluster_age=7.75,
               cutout_size=10,
               debug=True,
               verbose=True,
               no_lk_cache=True)

c.lightcurves_summary_file()
c.access_lightcurve(1)
