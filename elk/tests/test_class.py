import astropy.units as u
from elk.ensemble import EnsembleLC

c = EnsembleLC(output_path="../../output",
               identifier='NGC 419',
               location='23.58271, +61.1236',
               radius=.046,
               cluster_age=7.75,
               cutout_size=10,
               ignore_scattered_light=True,
               verbose=True,
               minimize_memory=True)

c.lightcurves_summary_file()

print(c.lcs)
