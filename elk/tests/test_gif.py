import elk
from elk.lightcurve import TESSCutLightcurve
import lightkurve as lk

search_results = lk.search_tesscut('NGC 7790')

print(search_results[1])

lc = TESSCutLightcurve(radius=2.9/60, lk_search_result=search_results[1], cutout_size=10,
                       save_pixel_periodograms=True, progress_bar=True)

lc.correct_lc()

print(lc.pixel_periodograms)