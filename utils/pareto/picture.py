import numpy
import pareto
import pandas
import matplotlib.figure
import matplotlib.backends.backend_agg as agg
import matplotlib.backends.backend_svg as svg

data = pandas.read_table("data.txt", header=None, names=['a','b'], sep=" ")
# data = pandas.read_table("data.txt", header=None, names=[0,1], sep=" ")

sets = {}
archives = {}

fig = matplotlib.figure.Figure(figsize=(15,15))
agg.FigureCanvasAgg(fig)

counter = 0
# print ("data", data)
# print ("columns", data.columns)
# print ("data.itertuples(False)", data.itertuples(False))
resolutions = [1e-9, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.4, 1.0]

# for row in data.itertuples(index=False):
#     print("row", row)

for resolution in resolutions:
    archives[resolution] = pareto.eps_sort([data.itertuples(False)], [0,1], [resolution]*2)
    #sets[resolution] = pandas.DataFrame(data=archives[resolution])


    sets[resolution] = pandas.DataFrame(data=archives[resolution].archive)


for resolution in resolutions:
    print ("resolution", resolution)
    counter += 1
    ax = fig.add_subplot(3,3,counter)
    ax.scatter(data['a'], data['b'], lw=0, facecolor=(0.7, 0.7, 0.7), zorder=-1)
    print ("sets[resolution]",sets[resolution])
    print ("sets[resolution]['a']", sets[resolution]['a'])
    print("sets[resolution]['b']", sets[resolution]['b'])
    ax.scatter(sets[resolution]['a'], sets[resolution]['b'], facecolor=(1.0, 1.0, 0.4), zorder=1, s=50)

    for box in archives[resolution].boxes:
        print ("box", box)
        ll = [box[0] * resolution, box[1] * resolution]
        print ("ll", ll)

        # make a rectangle in the Y direction
        rect = matplotlib.patches.Rectangle((ll[0], ll[1] + resolution), 1.4 - ll[0], 1.4 - ll[1], lw=0, facecolor=(1.0,0.8,0.8), zorder=-10)
        ax.add_patch(rect)

        # make a rectangle in the X direction
        rect = matplotlib.patches.Rectangle((ll[0] + resolution, ll[1]), 1.4 - ll[0], 1.4 - ll[1], lw=0, facecolor=(1.0,0.8,0.8), zorder=-10)
        ax.add_patch(rect)

    if resolution < 1e-3:
        spacing = 0.2
    else:
        spacing = resolution
        while spacing < 0.2:
            spacing *= 2

    ax.set_xticks(numpy.arange(0, 1.2, spacing))
    ax.set_yticks(numpy.arange(0, 1.2, spacing))
    
    if resolution > 0.001:
        ax.hlines(numpy.arange(0, 1.4, resolution), 0, 1.4, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)
        ax.vlines(numpy.arange(0, 1.4, resolution), 0, 1.4, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.2)
    ax.set_title("Epsilon resolution: {0:.2g}".format(resolution))
    ax.set_xlabel(r'$f_1$')
    ax.set_ylabel(r'$f_2$')



fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.savefig("variety")

fig = matplotlib.figure.Figure(figsize=(5,5))
agg.FigureCanvasAgg(fig)
resolution = 0.06
sets[resolution] = pandas.DataFrame(data=pareto.eps_sort([data.itertuples(False)], [0,1], [resolution]*2).archive)

ax = fig.add_subplot(1,1,1)
ax.scatter(data['a'], data['b'], lw=0, facecolor=(0.7, 0.7, 0.7), zorder=-1)
ax.scatter(sets[resolution]['a'], sets[resolution]['b'], facecolor=(1.0, 1.0, 0.4), zorder=1, s=50)

for box in archives[resolution].boxes:
    ll = [box[0] * resolution, box[1] * resolution]

    # make a rectangle in the Y direction
    rect = matplotlib.patches.Rectangle((ll[0], ll[1] + resolution), 1.4 - ll[0], 1.4 - ll[1], lw=0, facecolor=(1.0,0.8,0.8), zorder=-10)
    ax.add_patch(rect)

    # make a rectangle in the X direction
    rect = matplotlib.patches.Rectangle((ll[0] + resolution, ll[1]), 1.4 - ll[0], 1.4 - ll[1], lw=0, facecolor=(1.0,0.8,0.8), zorder=-10)
    ax.add_patch(rect)
if resolution < 1e-3:
    spacing = 0.2
else:
    spacing = resolution
    while spacing < 0.2:
        spacing *= 2

ax.set_xticks(numpy.arange(0, 1.2, spacing))
ax.set_yticks(numpy.arange(0, 1.2, spacing))

if resolution > 0.001:
    ax.hlines(numpy.arange(0, 1.4, resolution), 0, 1.4, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)
    ax.vlines(numpy.arange(0, 1.4, resolution), 0, 1.4, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)
ax.set_xlim(0, 1.2)
ax.set_ylim(0, 1.2)
ax.set_title("Epsilon resolution: {0:.2g}".format(resolution))
ax.set_xlabel(r'$f_1$')
ax.set_ylabel(r'$f_2$')

fig.savefig("example")
    

fig = matplotlib.figure.Figure(figsize=(5,5))
agg.FigureCanvasAgg(fig)

ax = fig.add_subplot(1,1,1)
ax.scatter(data['a'], data['b'], lw=0, facecolor=(0.7, 0.7, 0.7), zorder=-1)
spacing = 0.2
ax.set_xticks(numpy.arange(0, 1.2, spacing))
ax.set_yticks(numpy.arange(0, 1.2, spacing))
ax.set_xlim(0, 1.2)
ax.set_ylim(0, 1.2)
ax.set_title("Unsorted Data")
ax.set_xlabel(r'$f_1$')
ax.set_ylabel(r'$f_2$')

fig.savefig("unsorted")
