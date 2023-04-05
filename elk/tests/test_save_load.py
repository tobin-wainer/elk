import elk
import matplotlib.pyplot as plt

c = elk.ensemble.EnsembleLC(output_path="../../output",
               identifier='NGC 419',
               location='23.58271, +61.1236',
               radius=.046,
               cluster_age=7.75,
               cutout_size=10,
               debug=True,
               verbose=True,
               minimize_memory=True)

print(c.lightcurves_summary_file())

del c

c = elk.ensemble.from_fits("../../output/Corrected_LCs/NGC 419output_table.fits")
print(c)

for lc in c.lcs:
    print(elk.stats.J_stetson(lc.corrected_lc.time.value, lc.corrected_lc.flux.value, lc.corrected_lc.flux_err.value))
    plt.plot(lc.corrected_lc.time.value, lc.corrected_lc.flux.value)
    plt.show()

del c

c = elk.ensemble.EnsembleLC(output_path="../../output",
               identifier=='NGC 419',
               location='23.58271, +61.1236',
               radius=.046,
               cluster_age=7.75,
               cutout_size=10,
               debug=True,
               verbose=True,
               minimize_memory=True)

for lc in c.lcs:
    print(elk.stats.J_stetson(lc.corrected_lc.time.value, lc.corrected_lc.flux.value, lc.corrected_lc.flux_err.value))
    plt.plot(lc.corrected_lc.time.value, lc.corrected_lc.flux.value)
    plt.show()