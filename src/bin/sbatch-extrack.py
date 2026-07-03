#!/usr/bin/env python3
"""Script to submit ExTrack jobs to a SLURM cluster."""

import argparse
import getpass
import inspect
import os
import subprocess


extrack_prog = "run-extrack.py"


def _create_job_script(args: argparse.Namespace, prog_options: str, fn: str, fno: int) -> str:
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

    if not os.path.isfile(os.path.join(dir, extrack_prog)):
        raise Exception(f"Missing program: {os.path.join(dir, extrack_prog)}")
    if not os.path.isfile(fn):
        raise Exception(f"Missing data file: {fn}")

    # Job name uses PID to avoid script name clashes
    name = f"ext{fno}.{os.getpid()}"

    # Create the job file
    script = f"{name}.sh"
    with open(script, "w") as f:
        # job options
        # The -l option to bash is to make bash act as if a login shell (enables conda init)
        print(
            inspect.cleandoc(f"""\
      #!/bin/bash -l
      #SBATCH -J {name}
      #SBATCH -o "{os.path.splitext(fn)[0]}.{args.nb_states}.%j.log"
      #SBATCH --mail-user {args.username}@sussex.ac.uk
      #SBATCH --mail-type=END,FAIL
      #SBATCH --mem={args.memory}G
      #SBATCH --partition={args.partition}
      """),
            file=f,
        )
        if args.hours > 0:
            print(
                inspect.cleandoc(f"""\
        #SBATCH --time={args.hours}:00:00
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


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse the script arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Program to run ExTrack on a SLURM cluster.",
        epilog=inspect.cleandoc(f"""Note:

      This program makes assumptions on the installation of ExTrack and
      the run environment.

      Unknown arguments are parsed to {extrack_prog}."""),
    )
    parser.add_argument("data", nargs="+", help="Track file")
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
        "--hours",
        type=int,
        default=0,
        help="Optional maximum job hours (default: %(default)s)",
    )
    group.add_argument(
        "-p",
        "--partition",
        dest="partition",
        default="general",
        help="Job class, e.g. general; long (default: %(default)s)",
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

    return parser.parse_known_args()


if __name__ == "__main__":
    args, rest = _parse_args()

    # Number of states is special as it is used for the output filename.
    # This must be passed through manually.
    prog_options = ' '.join(rest) + f' --nb-states {args.nb_states}'

    for fno, fn in enumerate(args.data):
        script = _create_job_script(args, prog_options, fn, fno)

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
