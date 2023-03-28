from elk.lightcurve import TESSCutLightcurve
import lightkurve as lk
from time import time

search_results = lk.search_tesscut('NGC 7790')

print(search_results[1])

lc = TESSCutLightcurve(radius=2.9/60, lk_search_result=search_results[1], cutout_size=20,
                       save_pixel_periodograms=True, progress_bar=False)

lc.correct_lc()


start = time()
lc.diagnose_lc_periodogram('.', freq_bins='auto')
print("Runtime:", time() - start)
