#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Run untimed production observation: python -m infb_components.probes.observe_launch <server args>."""
import multiprocessing.spawn
import runpy

from infb_components.probes.observer import install


def main():
    original = multiprocessing.spawn.get_command_line
    def command_line(**kwargs):
        command = original(**kwargs)
        # SGLang starts TP workers with spawn. Install the same trusted observer
        # in each interpreter before it enters multiprocessing's worker entry.
        for index, argument in enumerate(command[:-1]):
            if argument in ("-c", b"-c"):
                code = command[index+1]
                prefix = "from infb_components.probes.observer import install; install(); "
                command[index+1] = prefix.encode() + code if isinstance(code, bytes) else prefix + code
                break
        return command
    multiprocessing.spawn.get_command_line = command_line
    install()
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
