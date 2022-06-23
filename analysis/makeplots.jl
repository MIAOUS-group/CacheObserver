using CSV
using Plots
pgfplotsx()

# Generals TODO : Fix the ticks, add legends
#eaps = [0,12,13,14]
eaps = [0,12,13,14,15]
len_eaps = length(eaps)
types = ["S","U"]
#types = ["S"]
levels = [0,1,2]

methods = ["SF", "SR", "FF"]

plot_lock = ReentrantLock()

slices_offset_0 = [0, 1, 2, 8, 14, 15, 30, 31, 32, 55, 56, 61, 62, 63]
#slices_offset_0 = []

diff_slices_offset_0 = [0, 1, 2, 61, 62, 63]

function make_name(eap, type, level)
    string("eap/eap-with-", eap, "-prefetcher.", type, level, ".csv")
end

all_file_names = fill((0,0,0,""), length(eaps), length(types), length(levels))
Threads.@threads for x in 1:len_eaps
    for (y,type) in enumerate(types)
        for (z,level) in enumerate(levels)
            all_file_names[x,y,z] = (x,y,z,make_name(eaps[x], type, level))
        end
    end
end


#files = Matrix(CSV, length(eaps), length(types), length(levels))
files = Array{Union{Nothing, Tuple{Int64,Int64,Int64,CSV.File}},3}(nothing, length(eaps), length(types), length(levels))

Threads.@threads for f in all_file_names
    x = f[1]
    y = f[2]
    z = f[3]
    name = f[4]
    files[x,y,z] = (x,y,z,CSV.File(name))
end

function graph_0(name, csv)
    data = [csv.Probe_FF_HR, csv.Probe_SR_HR, csv.Probe_SF_HR]
    x = range(0, 63)
    y = range(0, 2)
    function f(x, y)
        data[y + 1][x + 1]
    end
    lock(plot_lock) do
    graph = heatmap(x, y, f, yticks = ([0,1,2], ["FF", "SR", "SF"]), clims = (0, 1), xlabel="probe")
    savefig(graph, string("julia_", name, ".tikz"))
    savefig(graph, string("julia_", name, ".pdf"))
    end
end # Todo double check if something better can be done wrt y names ?

# TODO :
#
# - Split this function in a load data into square / cube structure and a plot function
# - Refactor the code below to compute the various squares / cubes and then do the plots.
# - Refactor the Slicing function too
# - Create a custom diagonal slice function ?


preamble_printed = false

push!(PGFPlotsX.CUSTOM_PREAMBLE,raw"\newcommand{\gdfigurewidth}{150mm}")
push!(PGFPlotsX.CUSTOM_PREAMBLE,raw"\newcommand{\gdfigureheight}{100mm}")

function graph2d(name, matrix, xlabel, ylabel)
    x = range(0, 63)
    y = range(0, 63)
    function hmp2d(x, y)
        matrix[x + 1, y + 1]
    end
    lock(plot_lock) do
        graph = heatmap(x, y, hmp2d, minorgrid=true, height = raw"{\gdfigureheight}}, width = {{\gdfigurewidth}", xlabel = xlabel, ylabel = ylabel, c = :blues, extra_kwargs =:subplot)
        if !preamble_printed
            global preamble_printed = true
            print(Plots.pgfx_preamble(graph))
        end
        savefig(graph, string(name, ".tikz"))
        savefig(graph, string(name, ".pdf"))
    end
end

function graph2dclims(name, matrix, clims, xlabel, ylabel)
    x = range(0, 63)
    y = range(0, 63)
    function hmp2d(x, y)
        matrix[x + 1, y + 1]
    end
    lock(plot_lock) do
        graph = heatmap(x, y, hmp2d, clims = clims, minorgrid=true, height = raw"{\gdfigureheight}}, width = {{\gdfigurewidth}", xlabel = xlabel, ylabel = ylabel, extra_kwargs =:subplot)
        savefig(graph, string(name, ".tikz"))
        savefig(graph, string(name, ".pdf"))
    end
end


function graph_1(basename, csv)
    # define the 2D arrays for the 3 heatmaps
    sf_probe_heatmap = fill(-1.0, 64, 64)
    sr_probe_heatmap = fill(-1.0, 64, 64)
    ff_probe_heatmap = fill(-1.0, 64, 64)
    # define 3 1D arrays to build the heatmap for average time of the first access in FF/SR and SF modes
    sf_offset_hit_time = fill(-1.0, 64)
    sr_offset_hit_time = fill(-1.0, 64)
    ff_offset_hit_time = fill(-1.0, 64)
    # iterates on the rows and fill in the 2D arrays.
    for row in csv
        offset = row.Offset_0
        probe = row.ProbeAddr
        @assert sf_probe_heatmap[offset+1,probe+1] == -1.0
        sf_probe_heatmap[offset + 1, probe + 1] = row.Probe_SF_HR
        sr_probe_heatmap[offset + 1, probe + 1] = row.Probe_SR_HR
        ff_probe_heatmap[offset + 1, probe + 1] = row.Probe_FF_HR

        if probe == 0
            @assert sf_offset_hit_time[offset + 1] == -1.0
            sf_offset_hit_time[offset + 1] = 0.0
        end
        sf_offset_hit_time[offset + 1] += row.Offset_0_SF_HR
        sr_offset_hit_time[offset + 1] += row.Offset_0_SR_HR
        ff_offset_hit_time[offset + 1] += row.Offset_0_FF_HR
        if probe == 63
            sf_offset_hit_time[offset + 1] /= 64
            sr_offset_hit_time[offset + 1] /= 64
            ff_offset_hit_time[offset + 1] /= 64
        end
    end

    graph2dclims(string("julia_", basename, "_SF"), sf_probe_heatmap, (0,1), "i", "probe")
    graph2dclims(string("julia_", basename, "_SR"), sr_probe_heatmap, (0,1), "i", "probe")
    graph2dclims(string("julia_", basename, "_FF"), ff_probe_heatmap, (0,1), "i", "probe")

    data = [ff_offset_hit_time, sr_offset_hit_time, sf_offset_hit_time]
    x = range(0, 63)
    y = range(0, 2)
    function f(x, y)
        data[y + 1][x + 1]
    end
    lock(plot_lock) do
        graph = heatmap(x, y, f)
        savefig(graph, string("julia_", basename, "_Offset_0_HT.tikz"))
        savefig(graph, string("julia_", basename, "_Offset_0_HT.pdf"))
    end
end

function myfill(element, dimensions)
    res = fill(element, dimensions)
    res = map(x -> deepcopy(x), res)
    res
end

function cube_flatten_z(cubes)
    len = length(cubes)
    res = myfill(myfill(0.0,(64,64)), len)
    for k in range(1,64)
        Threads.@threads for i in range(1,64)
            for j in range(1,64)
                for l in range(1,len)
                    res[l][i,j] += cubes[l][i,j,k]
                end
            end
        end
    end
    res
end

function slice_extract_x(cubes, slices)
    slice_length = length(slices)
    cube_length = length(cubes)
    res = myfill(myfill(myfill(0.0, (64, 64)), slice_length), cube_length)
    for i in range(1,64)
        for j in range(1,64)
            for (k,slice) in enumerate(slices)
                for l in range(1, cube_length)
                    res[l][k][i, j] = cubes[l][slice+1, i, j]
                end
            end
        end
    end
    res
end

function graph_2(basename, csv)
    # First define a 3D cube for the resulting data ?
    sf_probe_heatmap = myfill(-1.0, (64, 64, 64))
    sr_probe_heatmap = myfill(-1.0, (64, 64, 64))
    ff_probe_heatmap = myfill(-1.0, (64, 64, 64))
    # Fill in the 3D cube, then create the various slices and flattenings
    # Flattened Cube with x = first addr, y = second addr, compute the sum of prefetches ?
    # Grab a few random first adresses and look at them with x = second addr, y = probe addr
    # 0,1, 62,63 14, 15 plus one other depending on what appears

    # Also define and fill in a 2D matrix of offset1-offset2 hit time.
    sf_offset_hit_time = myfill(-1.0, (64, 64))
    sr_offset_hit_time = myfill(-1.0, (64, 64))
    ff_offset_hit_time = myfill(-1.0, (64, 64))

    for row in csv
        probe = row.ProbeAddr
        offset_0 = row.Offset_0
        offset_1 = row.Offset_1
        @assert sf_probe_heatmap[offset_0 + 1, offset_1 + 1, probe + 1] == -1.0
        sf_probe_heatmap[offset_0 + 1, offset_1 + 1, probe + 1] = row.Probe_SF_HR
        sr_probe_heatmap[offset_0 + 1, offset_1 + 1, probe + 1] = row.Probe_SR_HR
        ff_probe_heatmap[offset_0 + 1, offset_1 + 1, probe + 1] = row.Probe_FF_HR

        if probe == 0
            @assert sf_offset_hit_time[offset_0 + 1, offset_1 + 1] == -1.0
            sf_offset_hit_time[offset_0 + 1, offset_1 + 1] = 0.0
        end
        sf_offset_hit_time[offset_0 + 1, offset_1 + 1] += row.Offset_1_SF_HR
        sr_offset_hit_time[offset_0 + 1, offset_1 + 1] += row.Offset_1_SR_HR
        ff_offset_hit_time[offset_0 + 1, offset_1 + 1] += row.Offset_1_FF_HR
        if probe == 63
            sf_offset_hit_time[offset_0 + 1, offset_1 + 1] /= 64
            sr_offset_hit_time[offset_0 + 1, offset_1 + 1] /= 64
            ff_offset_hit_time[offset_0 + 1, offset_1 + 1] /= 64
        end

    end
    allprobes = cube_flatten_z([sf_probe_heatmap, sr_probe_heatmap, ff_probe_heatmap])
    sf_probe_heatmap_allprobes = allprobes[1]
    sr_probe_heatmap_allprobes = allprobes[2]
    ff_probe_heatmap_allprobes = allprobes[3]



    all_slices = slice_extract_x([sf_probe_heatmap, sr_probe_heatmap, ff_probe_heatmap], slices_offset_0)
    sf_probe_slices_heatmaps = all_slices[1]
    sr_probe_slices_heatmaps = all_slices[2]
    ff_probe_slices_heatmaps = all_slices[3]

    graph2d(string("julia_", basename, "_SF_AllProbes"), sf_probe_heatmap_allprobes, "i", "j")
    graph2d(string("julia_", basename, "_SR_AllProbes"), sr_probe_heatmap_allprobes, "i", "j")
    graph2d(string("julia_", basename, "_FF_AllProbes"), ff_probe_heatmap_allprobes, "i", "j")


    for (i, offset_0) in enumerate(slices_offset_0)
        print(offset_0)
        data = sf_probe_slices_heatmaps[i]
        graph2dclims(string("julia_", basename, "_SF_Slice_", offset_0),sf_probe_slices_heatmaps[i],(0,1), "j", "probe")
        graph2dclims(string("julia_", basename, "_SR_Slice_", offset_0),sr_probe_slices_heatmaps[i],(0,1), "j", "probe")
        graph2dclims(string("julia_", basename, "_FF_Slice_", offset_0),ff_probe_slices_heatmaps[i],(0,1), "j", "probe")
    end
    [sf_probe_heatmap, sr_probe_heatmap, ff_probe_heatmap]
end

Threads.@threads for file in files[:,:,1]
    name = string("eap_",eaps[file[1]],"_",types[file[2]],levels[file[3]])
    graph_0(name, file[4])
    print(string(name,"\n"))
end

Threads.@threads for file in files[:,:,2]
    name = string("eap_",eaps[file[1]],"_",types[file[2]],levels[file[3]])
    graph_1(name, file[4])
    print(string(name,"\n"))
end


cubes = fill(0.0, length(eaps), length(types), 3, 64, 64, 64)

Threads.@threads for file in files[:,:,3]
    name = string("eap_",eaps[file[1]],"_",types[file[2]],levels[file[3]])
    (sf,sr,ff) = graph_2(name, file[4])
    cubes[file[1], file[2], 1, :, :, :] = sf
    cubes[file[1], file[2], 2, :, :, :] = sr
    cubes[file[1], file[2], 3, :, :, :] = ff
    print(string(name,"\n"))
end




print("Computing 14 union 13...")

function cube_max(cubes_1, cubes_2)
    @assert size(cubes_1) == size(cubes_2)
    sizes = size(cubes_1)
    @assert length(sizes) == 5
    res = fill(0.0, sizes)
    for i in range(1,sizes[1])
        for j in range(1,sizes[2])
            Threads.@threads for k in range(1,64)
                for l in range(1, 64)
                    for m in range(1, 64)
                        res[i,j,k,l,m] = max(cubes_1[i,j,k,l,m], cubes_2[i,j,k,l,m])
                    end
                end
            end
        end
    end
    res
end

index_0  = findfirst(isequal(0),  eaps)
index_12 = findfirst(isequal(12), eaps)
index_13 = findfirst(isequal(13), eaps)
index_14 = findfirst(isequal(14), eaps)

cube_max_13_14 = cube_max(cubes[index_13,:,:,:,:,:], cubes[index_14,:,:,:,:,:])

function do_cubes(name, cubes)
    cube_list = []
    index_list = []
    for type in range(1,length(types))
        for method in range(1,3)
            push!(cube_list, cubes[type,method,:,:,:])
            push!(index_list, (type, method))
        end
    end
    allgraphs = cube_flatten_z(cube_list)
    for (i,(type,method)) in enumerate(index_list)
        graph2d(string(name, "_", types[type], "2_", methods[method], "_AllProbes"), allgraphs[i], "i", "j")
        for slice in diff_slices_offset_0
            graph2d(string(name,"_", types[type], "2_", methods[method], "_Slice_", slice), cubes[type, method, slice+1,:,:], "j", "probe")
        end
    end
end

graph_13_14 = @task begin
    do_cubes("julia_max_13_14", cube_max_13_14)
    cube_list = []
    index_list = []
    for type in range(1,length(types))
       for method in range(1,3)
           push!(cube_list, cube_max_13_14[type,method,:,:,:])
           push!(index_list, (type, method))
       end
    end
    allgraphs = cube_flatten_z(cube_list)
    for (i,(type,method)) in enumerate(index_list)
       graph2d(string("julia_max_13_14_", types[type], "2_", methods[method], "_AllProbes"), allgraphs[i], "i", "j")
    end
end
schedule(graph_13_14)


print(" OK\n")

print("Computing Any difference between 0 and 12...")

function cube_differences(cubes_1, cubes_2)
    @assert size(cubes_1) == size(cubes_2)
    sizes = size(cubes_1)
    @assert length(sizes) == 5
    res = fill(0.0, sizes)
    for i in range(1,sizes[1])
        for j in range(1,sizes[2])
            Threads.@threads for k in range(1,64)
                for l in range(1, 64)
                    for m in range(1, 64)
                        res[i,j,k,l,m] = abs(cubes_1[i,j,k,l,m] - cubes_2[i,j,k,l,m])
                    end
                end
            end
        end
    end
    res
end

cube_diff_0_12 = cube_differences(cubes[index_0,:,:,:,:,:], cubes[index_12,:,:,:,:,:])

graph_0_12 = @task begin
    do_cubes("julia_diff_0_12", cube_diff_0_12)
    cube_list = []
    index_list = []
    for type in range(1,length(types))
       for method in range(1,3)
           push!(cube_list, cube_diff_0_12[type,method,:,:,:])
           push!(index_list, (type, method))
       end
    end
    allgraphs = cube_flatten_z(cube_list)
    for (i,(type,method)) in enumerate(index_list)
       graph2d(string("julia_diff_0_12_", types[type], "2_", methods[method], "_AllProbes"), allgraphs[i], "i", "j")
    end
end
schedule(graph_0_12)

print(" OK\n")



print("Computing Differences between 12 and (13 union 14)...")

cube_diff_12_1314 = cube_differences(cubes[index_0,:,:,:,:,:], cube_max_13_14)

graph_12_1314 = @task begin
    do_cubes("julia_diff_12_1314", cube_diff_12_1314)
    cube_list = []
    index_list = []
    for type in range(1,length(types))
       for method in range(1,3)
           push!(cube_list, cube_diff_12_1314[type,method,:,:,:])
           push!(index_list, (type, method))
       end
    end
    allgraphs = cube_flatten_z(cube_list)
    for (i,(type,method)) in enumerate(index_list)
       graph2d(string("julia_diff_12_1314", types[type], "2_", methods[method], "_AllProbes"), allgraphs[i], "i", "j")
       for slice in diff_slices_offset_0
       end
    end
end
schedule(graph_12_1314)

wait(graph_13_14)
wait(graph_0_12)
wait(graph_12_1314)
print("done\n")
