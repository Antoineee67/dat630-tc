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
        sub_df = df.query("nbody==@nbody").sort_values("runtime")
        arr = sub_df["mpi_depth"].to_numpy()

        
        best_depth.append(meanCal(arr, averageSize))

    fig, ax = plt.subplots()
    ax.plot(nbody_sizes, best_depth)
    ax.set_ylabel("MPI Depth", fontsize=16)
    ax.set_xlabel("$N$", fontsize = 16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    loc = plticker.MultipleLocator(base=100000) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    fig.savefig(f"./plots/mpi-optimal_depth", dpi=300)

def openmp_threashold(df: pandas.DataFrame):
    nbody_sizes = df["nbody"].unique()
    nbody_sizes.sort()
    fig, ax = plt.subplots()
    for nbody in nbody_sizes:
        sub_df = df.query("nbody==@nbody")
        ax.plot(sub_df["omp_threashold"], sub_df["runtime"], label=f"${nbody/100000}\\times 10^5$")

    
    
    ax.set_ylabel("Runtime", fontsize=16)
    ax.set_xlabel("OpenMP Threashold", fontsize = 16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    handles, labels = ax.get_legend_handles_labels()
    labels[-1] = '$10.0\\times 10^5~~$  ' #Ugly fix.
    print(labels)
    ax.legend(handles[::-1], labels[::-1], loc="center right", bbox_to_anchor=(1.4, 0.5), frameon=False, title="$N$")
    fig.savefig(f"./plots/omp-threashold", dpi=300)



if not os.path.exists("./plots"):
    os.makedirs("./plots")


df_mpi_depth = pandas.read_csv(r"../localjobs/mpi_depth_data.txt",header=0, decimal=".", delimiter="\t")
openMP_depth = pandas.read_csv(r"../localjobs/openMP_threshold_data.txt",header=0, decimal=".", delimiter="\t")


mpi_depth_optimum_depth(df_mpi_depth, 2)

openmp_threashold(openMP_depth)

