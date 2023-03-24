import elk
from elk.lightcurve import TESSCutLightcurve
import lightkurve as lk
import numpy as np

search_results = lk.search_tesscut('NGC 7790')

print(search_results[1])

lc = TESSCutLightcurve(radius=2.9/60, lk_search_result=search_results[1], cutout_size=20,
                       save_pixel_periodograms=True, progress_bar=True)

lc.correct_lc()

lc.make_periodogram_peak_pixels_gif('.', freq_bins=np.logspace(-1, 1, 20))
