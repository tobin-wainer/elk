from elk.lightcurve import TESSCutLightcurve
import lightkurve as lk
from time import time

search_results = lk.search_tesscut('NGC 7790')

print(search_results[1])

lc = TESSCutLightcurve(radius=2.9/60, lk_search_result=search_results[1], cutout_size=40,
                       save_pixel_periodograms=True, progress_bar=False)

lc.correct_lc()


start = time()
lc.make_periodogram_peak_pixels_gif('.', freq_bins='auto')
print("Runtime:", time() - start)
