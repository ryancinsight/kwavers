# ============================================================================
# k-wave-julia 3-D phased-array parity driver
# ============================================================================
# Drives a flat linear phased-array transducer (no focus, no steering) with
# a tone burst, captures the peak pressure on a y-z slice plane downstream
# of the array, and writes both the slice and the binary mask out for the
# Python comparator.
#
# A flat array with no focus / no steering is mathematically a binary
# mask source plane driven with one common signal, which is what the Python
# side of this parity reproduces in pykwavers without needing KWaveArray.
# ============================================================================
using DelimitedFiles
using JSON
using KWave

function parse_args(args)
    values = Dict{String, String}()
    i = 1
    while i <= length(args)
        key = args[i]
        if !startswith(key, "--") || i == length(args)
            error("Expected --key value pair, got '$(key)'")
        end
        values[key[3:end]] = args[i + 1]
        i += 2
    end
    return values
end

function main()
    args = parse_args(ARGS)
    nx       = parse(Int, args["nx"])
    ny       = parse(Int, args["ny"])
    nz       = parse(Int, args["nz"])
    dx       = parse(Float64, args["dx"])
    c0       = parse(Float64, args["c0"])
    rho0     = parse(Float64, args["rho0"])
    n_elem   = parse(Int, args["n-elements"])
    elem_w   = parse(Int, args["element-width"])
    elem_l   = parse(Int, args["element-length"])
    pos_x    = parse(Int, args["pos-x-1based"])
    pos_y    = parse(Int, args["pos-y-1based"])
    pos_z    = parse(Int, args["pos-z-1based"])
    src_freq = parse(Float64, args["src-freq"])
    n_cycles = parse(Int, args["n-cycles"])
    src_peak = parse(Float64, args["src-peak"])
    pml_size = parse(Int, get(args, "pml-size", "10"))
    sensor_x = parse(Int, args["sensor-x-1based"])

    out_pmax = args["out-pmax-csv"]
    out_mask = args["out-mask-csv"]
    out_meta = args["out-meta"]

    kgrid = KWaveGrid(nx, dx, ny, dx, nz, dx)
    make_time!(kgrid, c0)
    nt = kgrid.Nt[]
    dt = kgrid.dt[]

    medium = KWaveMedium(sound_speed = c0, density = rho0)

    # Build the transducer mask deterministically (no KWaveTransducer
    # convenience layer — we want the SAME source to be reproducible by
    # pykwavers as a plain mask).
    mask = falses(nx, ny, nz)
    elem_pitch = elem_w  # element_spacing = 0
    for k in 1:n_elem
        y_start = pos_y + (k - 1) * elem_pitch
        y_end   = y_start + elem_w - 1
        z_start = pos_z
        z_end   = z_start + elem_l - 1
        if y_end <= ny && z_end <= nz
            mask[pos_x, y_start:y_end, z_start:z_end] .= true
        end
    end

    n_active = count(mask)
    println("[julia] transducer: $(n_elem) elements, $(n_active) active grid points")

    # Tone burst: n_cycles at src_freq, Hann-tapered envelope, peak src_peak.
    t_axis = (0:nt-1) .* dt
    burst_dur = n_cycles / src_freq
    sig = zeros(nt)
    for k in 1:nt
        if t_axis[k] < burst_dur
            env = 0.5 * (1.0 - cos(2π * t_axis[k] / burst_dur))
            sig[k] = src_peak * env * sin(2π * src_freq * t_axis[k])
        end
    end

    # Pressure source on the transducer mask — same signal applied to every
    # active grid point.
    p_signal_mat = repeat(sig', n_active, 1)
    source = KWaveSource(p_mask = mask, p = p_signal_mat)

    # Sensor: y-z plane at i = sensor_x. Record p_max.
    sensor_mask = falses(nx, ny, nz)
    sensor_mask[sensor_x, :, :] .= true
    sensor = KWaveSensor(mask = sensor_mask, record = [:p_max])

    println("[julia] forward kspace_first_order nx=$nx ny=$ny nz=$nz nt=$nt")
    fwd = kspace_first_order(
        kgrid, medium, source, sensor;
        pml_size = pml_size, plot_sim = false,
    )

    p_max_raw = fwd[:p_max]
    pmax_slice = ndims(p_max_raw) == 1 ? reshape(p_max_raw, ny, nz) : p_max_raw

    open(out_pmax, "w") do io
        for k in 1:size(pmax_slice, 2)
            write(io, join(pmax_slice[:, k], ","))
            write(io, "\n")
        end
    end

    # Save mask in column-major order so pykwavers can ingest the same.
    open(out_mask, "w") do io
        for k in 1:nz, j in 1:ny, i in 1:nx
            if mask[i, j, k]
                write(io, "$(i),$(j),$(k)\n")
            end
        end
    end

    open(out_meta, "w") do io
        JSON.print(io, Dict(
            "nx"=>nx,"ny"=>ny,"nz"=>nz,"dx"=>dx,"nt"=>nt,"dt"=>dt,
            "c0"=>c0,"rho0"=>rho0,"n_elements"=>n_elem,
            "element_width"=>elem_w,"element_length"=>elem_l,
            "pos"=>[pos_x,pos_y,pos_z],
            "src_freq"=>src_freq,"n_cycles"=>n_cycles,"src_peak"=>src_peak,
            "pml_size"=>pml_size,"sensor_x_1based"=>sensor_x,
            "n_active_grid_points"=>n_active,
            "pmax_peak"=>maximum(pmax_slice),
            "engine"=>"KWave.jl/kspace_first_order pressure-mask transducer",
        ))
    end
    println("[julia] done: peak |p_max| on slice = ", maximum(pmax_slice))
end

main()
