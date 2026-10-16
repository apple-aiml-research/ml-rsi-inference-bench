#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Launch the declared serving recipe: python -m gpt_oss.launch [--print]."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys


def command(serving_class, *, host='127.0.0.1', port=8000):
    recipes=json.loads(Path(__file__).with_name('recipes.json').read_text())
    return [sys.executable,'-m','sglang.launch_server',
            '--model-path',recipes['model'],*recipes[serving_class],
            '--host',host,'--port',str(port)]


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--class',dest='serving_class',choices=('S0',),default='S0')
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=8000)
    p.add_argument('--print',action='store_true',dest='print_only')
    a=p.parse_args()
    argv=command(a.serving_class,host=a.host,port=a.port)
    if a.print_only:
        print(shlex.join(argv))
    else:
        os.execv(sys.executable,argv)
