
# test_run.py
#
# Robust runner for your ride-sharing simulation.
# - Tries to import Simulation (or alias older class names) from final_simulation.py
# - Ensures a default map exists
# - Runs the sim and writes a PNG, optionally opens it

import argparse
import os
import sys
import webbrowser

# ------- Flexible import shim -------
SimClass = None
try:
    from final_simulation import Simulation as SimClass
except Exception:
    try:
        from final_simulation import CompleteSimulation as SimClass
    except Exception:
        try:
            from final_simulation import FinalIntegratedSimulation as SimClass
        except Exception:
            SimClass = None

if SimClass is None:
    raise ImportError(
        "Could not find a simulation class in final_simulation.py. "
        "Define one of: `Simulation`, `CompleteSimulation`, or `FinalIntegratedSimulation` "
        "and export it from final_simulation.py."
    )

# ------- Defaults -------
DEFAULT_MAP = "map.csv"
DEFAULT_PNG = "simulation_summary.png"

DEFAULT_MAP_CONTENT = """start_node,end_node,travel_time
A,B,5
B,A,5
A,C,3
C,A,3
B,D,4
D,B,4
C,D,1
D,C,1
A,E,7
E,A,7
B,F,6
F,B,6
C,F,2
F,C,2
D,G,3
G,D,3
E,F,4
F,E,4
F,G,2
G,F,2
"""

def ensure_map(path: str) -> None:
    if os.path.exists(path):
        return
    with open(path, "w", newline="") as f:
        f.write(DEFAULT_MAP_CONTENT)
    print(f"[setup] Created default debug map at {path}")

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run the ride-sharing Simulation test harness.")
    p.add_argument("--max-time", type=float, default=100.0, help="Simulation horizon in seconds")
    p.add_argument("--num-cars", type=int, default=12, help="Fleet size")
    p.add_argument("--arrival-rate", type=float, default=2.0, help="Mean inter-arrival time (seconds)")
    p.add_argument("--map-file", type=str, default=DEFAULT_MAP, help="Path to map CSV")
    p.add_argument("--output", type=str, default=DEFAULT_PNG, help="PNG output path")
    p.add_argument("--realtime", action="store_true", help="Stream event log to console")
    p.add_argument("--surge", action="store_true", help="Enable simple zone-based surge")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--open", action="store_true", help="Open the PNG after completion")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    ensure_map(args.map_file)

    # Construct simulation regardless of the concrete class name
    sim = SimClass(
        max_time=args.max_time,
        num_cars=args.num_cars,
        mean_arrival=args.arrival_rate,
        map_file=args.map_file,
        png_path=args.output,
        realtime=args.realtime,
        surge=args.surge,
        seed=args.seed,
    )

    print(
        f"[run] max_time={args.max_time}  num_cars={args.num_cars}  "
        f"arrival={args.arrival_rate}  map='{args.map_file}'  png='{args.output}'  "
        f"realtime={args.realtime}  surge={args.surge}  seed={args.seed}"
    )

    metrics = sim.run()

    if metrics:
        print("\n=== FINAL METRICS ===")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.2f}")
            else:
                print(f"{k}: {v}")

    if os.path.exists(args.output):
        print(f"[ok] PNG written: {args.output}")
        if args.open:
            try:
                webbrowser.open(os.path.abspath(args.output))
            except Exception as e:
                print(f"[warn] Could not auto-open PNG: {e}")
    else:
        print("[warn] PNG was not created.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
