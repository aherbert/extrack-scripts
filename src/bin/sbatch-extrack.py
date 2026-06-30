#!/usr/bin/env python3
"""Script to submit ExTrack jobs to a SLURM cluster."""

import argparse
import getpass
import inspect
import os
import subprocess


def _create_job_script(args: argparse.Namespace, fn: str, fno: int) -> str:
    """Create the SLURM job script.

    Args:
        args: Program arguments
        fn: Track file
        fno: File number

    Returns:
        The name of the script file
    """
    # Validate installation
    dir = os.path.dirname(__file__)
    extrack_prog = "run-extrack.py"

    if not os.path.isfile(os.path.join(dir, extrack_prog)):
        raise Exception(f"Missing program: {os.path.join(dir, extrack_prog)}")
    if not os.path.isfile(fn):
        raise Exception(f"Missing data file: {fn}")

    # Job name uses PID to avoid script name clashes
    name = f"ext{fno}.{os.getpid()}"

    # Options
    prog_options = f"--nb-states {args.nb_states}"

    # Create the job file
    script = f"{name}.sh"
    with open(script, "w") as f:
        # job options
        # The -l option to bash is to make bash act as if a login shell (enables conda init)
        print(
            inspect.cleandoc(f"""\
      #!/bin/bash -l
      #SBATCH -J {name}
      #SBATCH -o {name}."%j".out
      #SBATCH --mail-user {args.username}@sussex.ac.uk
      #SBATCH --mail-type=END,FAIL
      #SBATCH --mem={args.memory}G
      """),
            file=f,
        )
        if args.threads > 1:
            # For python multiprocessing we require shared memory parallelization.
            # It spawns new processes that all have access to the main memory of a single machine.
            print(
                inspect.cleandoc(f"""\
        #SBATCH --ntasks={args.threads}
        #SBATCH --nodes=1
        #SBATCH --cpus-per-task=1
        """),
                file=f,
            )
        # job script

        print(
            inspect.cleandoc(
                f"""
        conda activate extrack
        export PATH=$PATH:{dir}
        run-extrack.py "{fn}" {prog_options}
        rm {script}
        """), file=f)

        return script


def _parse_args() -> argparse.Namespace:
    """Parse the script arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Program to run ExTrack on a SLURM cluster.",
        epilog=inspect.cleandoc("""Note:

      This program makes assumptions on the installation of ExTrack and
      the run environment."""),
    )
    parser.add_argument("data", nargs="+", help="Track fail")
    group = parser.add_argument_group("Job submission")
    group.add_argument(
        "-u",
        "--username",
        dest="username",
        default=getpass.getuser(),
        help="Username (default: %(default)s)",
    )
    group.add_argument(
        "-t",
        "--threads",
        type=int,
        dest="threads",
        default=16,
        help="Threads (default: %(default)s)",
    )
    group.add_argument(
        "--memory",
        type=int,
        dest="memory",
        default=32,
        help="Memory in Gb (default: %(default)s)",
    )
    group.add_argument(
        "--submit",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Disable this to create the script but not submit using sbatch (default: %(default)s)",
    )
    group = parser.add_argument_group("ExTrack overrides")
    group.add_argument(
        "--nb-states",
        type=int,
        default=2,
        help="number of states (default: %(default)s)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    for fno, fn in enumerate(args.data):
        script = _create_job_script(args, fn, fno)

        # job submission
        if args.submit:
            print(
                subprocess.run(
                    ["sbatch", script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                ).stdout
            )
