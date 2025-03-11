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


def mpi_depth_optimum_depth_and_exchanged_bodies(df: pandas.DataFrame, averageSize):
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
    fig.savefig(f"./plots/mpi-optimal_depth.png", dpi=300)

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
    fig.savefig(f"./plots/omp-threashold.png", dpi=300)


def speedup_benchmark_plots(df: pandas.DataFrame):
    nr_nodes = df["mpi_nodes"].unique()
    fig, ax = plt.subplots()
    global baseline
    baseline = df.query("mpi_nodes == 1 and omp_threads == 1")["runtime"].max()
    for node in nr_nodes:
        sub_df = df.query("mpi_nodes == @node")

        ordering = sub_df["omp_threads"].sort_values().index
        ax.plot(sub_df["omp_threads"][ordering], baseline/sub_df["runtime"][ordering], label=f"Zen4 nodes: {node}")
    
    ax.set_ylabel("Speedup", fontsize=16)
    ax.set_xlabel("OpenMP threads", fontsize=16)
    ax.set_xscale("log", base=2)
    ax.set_title("$N=10^6$", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.xaxis.set_major_formatter(plticker.LogFormatter(base=2))
    ax.legend()
    #fig.legend(loc=7, bbox_to_anchor=(1,0.3))
    fig.savefig(f"./plots/multinode_speedup_benchmark.png", dpi=300)

    sub_df = df.query("mpi_nodes == 1")
    fig, ax = plt.subplots()
    ordering = sub_df["omp_threads"].sort_values().index
    ax.plot(sub_df["omp_threads"][ordering], baseline/sub_df["runtime"][ordering], label=f"Zen4 nodes: {node}")
    ax.set_ylabel("Speedup", fontsize=16)
    ax.set_xlabel("OpenMP threads", fontsize=16)
    ax.set_xscale("log", base=2)
    ax.set_title("$N=10^6$", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.xaxis.set_major_formatter(plticker.LogFormatter(base=2))
    fig.savefig("./plots/single_node_speedup_benchmark.png", dpi=300)
    


def speedup_omp_levels(df: pandas.DataFrame):
    fig, ax = plt.subplots()
    global baseline
    print(baseline)
    ordering = df["runtime"].sort_values().index
    ax.plot(df["omp_max_active_levels"][ordering], baseline/df["runtime"][ordering])
    ax.set_ylabel("Speedup", fontsize=16)
    ax.set_xlabel("OpenMP nested levels", fontsize=16)
    ax.set_title("$N=10^6$", fontsize=16)
    ax.xaxis.set_major_locator(plticker.IndexLocator(1,0))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    fig.savefig("./plots/omp_levels_benchmark.png", dpi=300)


def mpi_speedup(df: pandas.DataFrame):
    fig, ax = plt.subplots()
    global baseline
    sub_df = df.query("omp_threads==1")

    ordering = sub_df["mpi_nodes"].sort_values().index
    ax.plot(sub_df["mpi_nodes"][ordering], baseline/sub_df["runtime"][ordering])
    
    ax.set_ylabel("Speedup", fontsize=16)
    ax.set_xlabel("MPI nodes", fontsize=16)
    ax.set_title("$N=10^6$", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.xaxis.set_major_locator(plticker.IndexLocator(1,0))
    #fig.legend(loc=7, bbox_to_anchor=(1,0.3))
    fig.savefig(f"./plots/mpi_1thread.png", dpi=300)


#Use a special baseline because this was run on tstop = 0.3125.
def cuda_benchmark(df: pandas.DataFrame, special_baseline):
    fig, ax = plt.subplots()
    nr_streams = df["cuda_streams"].unique()
    for stream in nr_streams:
        sub_df = df.query("cuda_streams==@stream")
        ordering = sub_df["cuda_size"].sort_values().index
        ax.scatter(sub_df["cuda_size"][ordering], special_baseline/sub_df["runtime"][ordering], label=f"Nr CUDA streams: {stream}")
    

    ax.set_ylabel("Speedup", fontsize=16)
    ax.set_xlabel("Dispatch Size", fontsize=16)
    ax.set_title("$N=10^6$", fontsize=16)
    ax.set_xscale("log", base=2)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.xaxis.set_major_formatter(plticker.ScalarFormatter())
    ax.legend()
    
    fig.savefig(f"./plots/cuda_bench.png", dpi=300)



if not os.path.exists("./plots"):
    os.makedirs("./plots")


#df_mpi_depth = pandas.read_csv(r"../localjobs/mpi_depth_data.txt",header=0, decimal=".", delimiter="\t")
#openMP_depth = pandas.read_csv(r"../localjobs/openMP_threshold_data.txt",header=0, decimal=".", delimiter="\t")


#mpi_depth_optimum_depth_and_exchanged_bodies(df_mpi_depth, 2)

#openmp_threashold(openMP_depth)


#runtime_bench = pandas.read_csv(r"../benchmarks/vera-1-mpi-runtime-no-ompnest",header=0, decimal=".", delimiter="\t")

#speedup_benchmark_plots(runtime_bench)

#levels_bench = pandas.read_csv(r"../levelsTest/treecodeBenchmark.txt",header=0, decimal=".", delimiter="\t")

#speedup_omp_levels(levels_bench)

#mpi_speedup(runtime_bench)

cuda_bench = pandas.read_csv(r"../benchmarks/CudaBenchmark.txt",header=0, decimal=".", delimiter="\t")


cuda_benchmark(cuda_bench, 250/3)