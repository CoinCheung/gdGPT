# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed
import subprocess
import argparse
from .ops.op_builder.all_ops import ALL_OPS
from .git_version_info import installed_ops, torch_info
from deepspeed.accelerator import get_accelerator

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
END = '\033[0m'
SUCCESS = f"{GREEN} [SUCCESS] {END}"
OKAY = f"{GREEN}[OKAY]{END}"
WARNING = f"{YELLOW}[WARNING]{END}"
FAIL = f'{RED}[FAIL]{END}'
INFO = '[INFO]'

color_len = len(GREEN) + len(END)
okay = f"{GREEN}[OKAY]{END}"
warning = f"{YELLOW}[WARNING]{END}"


def op_report(verbose=True):
    max_dots = 23
    max_dots2 = 11
    h = ["op name", "installed", "compatible"]
    print("-" * (max_dots + max_dots2 + len(h[0]) + len(h[1])))
    print("DeepSpeed C++/CUDA extension op report")
    print("-" * (max_dots + max_dots2 + len(h[0]) + len(h[1])))

    print("NOTE: Ops not installed will be just-in-time (JIT) compiled at\n"
          "      runtime if needed. Op compatibility means that your system\n"
          "      meet the required dependencies to JIT install the op.")

    print("-" * (max_dots + max_dots2 + len(h[0]) + len(h[1])))
    print("JIT compiled ops requires ninja")
    ninja_status = OKAY if ninja_installed() else FAIL
    print('ninja', "." * (max_dots - 5), ninja_status)
    print("-" * (max_dots + max_dots2 + len(h[0]) + len(h[1])))
    print(h[0], "." * (max_dots - len(h[0])), h[1], "." * (max_dots2 - len(h[1])), h[2])
    print("-" * (max_dots + max_dots2 + len(h[0]) + len(h[1])))
    installed = f"{GREEN}[YES]{END}"
    no = f"{YELLOW}[NO]{END}"
    for op_name, builder in ALL_OPS.items():
        dots = "." * (max_dots - len(op_name))
        is_compatible = OKAY if builder.is_compatible(verbose) else no
        is_installed = installed if installed_ops.get(op_name, False) else no
        dots2 = '.' * ((len(h[1]) + (max_dots2 - len(h[1]))) - (len(is_installed) - color_len))
        print(op_name, dots, is_installed, dots2, is_compatible)
    print("-" * (max_dots + max_dots2 + len(h[0]) + len(h[1])))


def ninja_installed():
    try:
        import ninja  # noqa: F401
    except ImportError:
        return False
    return True


def nvcc_version():
    import torch.utils.cpp_extension
    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    if cuda_home is None:
        return f"{RED} [FAIL] cannot find CUDA_HOME via torch.utils.cpp_extension.CUDA_HOME={torch.utils.cpp_extension.CUDA_HOME} {END}"
    try:
        output = subprocess.check_output([cuda_home + "/bin/nvcc", "-V"], universal_newlines=True)
    except FileNotFoundError:
        return f"{RED} [FAIL] nvcc missing {END}"
    output_split = output.split()
    release_idx = output_split.index("release")
    release = output_split[release_idx + 1].replace(',', '').split(".")
    return ".".join(release)


def debug_report():
    max_dots = 33

    report = [("torch install path", torch.__path__), ("torch version", torch.__version__),
              ("deepspeed install path", deepspeed.__path__),
              ("deepspeed info", f"{deepspeed.__version__}, {deepspeed.__git_hash__}, {deepspeed.__git_branch__}")]
    if get_accelerator().device_name() == 'cuda':
        hip_version = getattr(torch.version, "hip", None)
        report.extend([("torch cuda version", torch.version.cuda), ("torch hip version", hip_version),
                       ("nvcc version", (None if hip_version else nvcc_version())),
                       ("deepspeed wheel compiled w.", f"torch {torch_info['version']}, " +
                        (f"hip {torch_info['hip_version']}" if hip_version else f"cuda {torch_info['cuda_version']}"))
                       ])
    else:
        report.extend([("deepspeed wheel compiled w.", f"torch {torch_info['version']} ")])

    print("DeepSpeed general environment info:")
    for name, value in report:
        print(name, "." * (max_dots - len(name)), value)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hide_operator_status',
                        action='store_true',
                        help='Suppress display of installation and compatibility statuses of DeepSpeed operators. ')
    parser.add_argument('--hide_errors_and_warnings', action='store_true', help='Suppress warning and error messages.')
    args = parser.parse_args()
    return args


def main(hide_operator_status=False, hide_errors_and_warnings=False):
    if not hide_operator_status:
        op_report(verbose=not hide_errors_and_warnings)
    debug_report()


def cli_main():
    args = parse_arguments()
    main(hide_operator_status=args.hide_operator_status, hide_errors_and_warnings=args.hide_errors_and_warnings)


if __name__ == "__main__":
    main()
