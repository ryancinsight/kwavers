import numpy as np
import pykwavers as kw
import sys
import traceback

def main():
    with open("examples/diag_vel.log", "w") as f:
        try:
            Nx, Ny, Nz = 88, 108, 44
            dx = 40e-3 / 128
            data = np.load("kw_results.npz")
            kw_dt = float(data["dt"])
            sig = data["input_signal"].flatten()
            num_ts = int(40e-6 / kw_dt)
            padded = np.zeros(num_ts)
            vl = min(len(sig), num_ts)
            padded[:vl] = sig[:vl]
            f.write(f"padded len={len(padded)}, ts={num_ts}, max={np.max(np.abs(padded)):.4e}\n")
            f.flush()
            
            grid = kw.Grid(nx=Nx, ny=Ny, nz=Nz, dx=dx, dy=dx, dz=dx)
            med = kw.Medium.homogeneous(sound_speed=1540.0, density=1000.0)
            sm = np.zeros((Nx, Ny, Nz))
            sm[1, 54, 22] = 1.0
            src = kw.Source.from_velocity_mask(sm, ux=padded, mode="dirichlet")
            f.write(f"Source type={src.source_type}, mode={src.source_mode}\n")
            f.flush()
            
            smask = np.zeros((Nx, Ny, Nz), dtype=bool)
            smask[10, 54, 22] = True
            sen = kw.Sensor.from_mask(smask)
            sim = kw.Simulation(grid, med, src, sen, solver=kw.SolverType.PSTD)
            f.write("Calling sim.run...\n")
            f.flush()
            
            r = sim.run(time_steps=num_ts, dt=kw_dt)
            f.write("sim.run returned\n")
            f.flush()
            
            sd = r.sensor_data
            f.write(f"sd type={type(sd)}, sd is None={sd is None}\n")
            if sd is not None:
                f.write(f"shape={sd.shape}, max={np.max(sd):.4e}, min={np.min(sd):.4e}\n")
            else:
                f.write("NO SENSOR DATA\n")
            f.flush()
        except Exception as e:
            f.write(f"ERROR: {e}\n")
            traceback.print_exc(file=f)
            f.flush()

if __name__ == "__main__":
    main()
