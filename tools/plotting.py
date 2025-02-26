import pandas
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import os
import seaborn as sns
sns.set_theme()
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams["axes.formatter.use_mathtext"] = True


def meanCal(arr, length):
    sum = 0
    for i in range(length):
        sum += arr[i]

    return sum/length


def mpi_depth_optimum_depth(df: pandas.DataFrame, averageSize):
    #averageSize is how many samples should be averaged to get the mpi_depth when sorted by runtime. 
    #Should probably be equal to how many samples you have for each initial condition.
    
    
    nbody_sizes = df["nbody"].unique()
    nbody_sizes.sort()
    best_depth = list()

    for nbody in nbody_sizes:
        sub_df = df_mpi_depth.query("nbody==@nbody").sort_values("runtime")
        arr = sub_df["mpi_depth"].to_numpy()

        
        best_depth.append(meanCal(arr, averageSize))

    fig, ax = plt.subplots()
    ax.plot(nbody_sizes, best_depth)
    ax.set_ylabel("Optimal Depth")
    ax.set_xlabel("$N$")
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    loc = plticker.MultipleLocator(base=100000) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    fig.savefig(f"./plots/optimal_depth", dpi=300)


if not os.path.exists("./plots"):
    os.makedirs("./plots")


df_mpi_depth = pandas.read_csv(r"../localjobs/mpi_depth_data.txt",header=0, decimal=".", delimiter="\t")

mpi_depth_optimum_depth(df_mpi_depth, 2)

