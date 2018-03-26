import holoviews as hv
import numpy as np
import holoviews.plotting.mpl

#renderer = hv.Store.renderers['matplotlib']
renderer = hv.renderer('matplotlib')#.instance(fig='svg', holomap='gif')

frequencies = [0.5, 0.75, 1.0, 1.25]

def sine_curve(phase, freq):
    xvals = [0.1* i for i in range(100)]
    return hv.Curve((xvals, [np.sin(phase+freq*x) for x in xvals]))

curve_dict = {f:sine_curve(0,f) for f in frequencies}

hmap = hv.HoloMap(curve_dict, kdims='frequency')
widget = renderer.get_widget(hmap, 'widgets')
#renderer.show(hmap)
renderer.show(widget)
