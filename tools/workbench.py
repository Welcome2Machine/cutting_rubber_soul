"""Local HTML workbench and portable result export; no web build dependencies."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import webbrowser

from run_support import ROOT, background_child_options


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--background',action='store_true',help='Start a hidden local server independent of this terminal')
    parser.add_argument('--open',action='store_true',help='Open the local UI in the default browser')
    parser.add_argument('--export',type=Path,help='Export an existing case directory instead of serving the UI')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--frames',type=int,default=11)
    parser.add_argument('--movie',action='store_true')
    args=parser.parse_args()
    if args.export:
        if not args.output:parser.error('--export requires --output')
        from workbench.results import export_case
        export_case(args.export,args.output,max_frames=args.frames,movie=args.movie)
        print(args.output/'result.bundle.json');return
    if not 1024<=args.port<=65535:parser.error('Use a local port in [1024,65535]')
    url=f'http://127.0.0.1:{args.port}/'
    if args.background:
        import urllib.request
        def ready():
            try:
                with urllib.request.urlopen(url+'api/catalog',timeout=1) as r:
                    return b'softcut-config/1' in r.read()
            except OSError:return False
        if not ready():
            folder=ROOT/'outputs/workbench';folder.mkdir(parents=True,exist_ok=True)
            with (folder/'server.log').open('ab') as log:
                child=subprocess.Popen([sys.executable,'-X','utf8','-u','-m','tools.workbench','--port',str(args.port)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,**background_child_options())
            for _ in range(50):
                if ready():break
                if child.poll() is not None:raise RuntimeError('Workbench failed to start; see outputs/workbench/server.log')
                time.sleep(.1)
            else:raise RuntimeError('Workbench startup not confirmed; see outputs/workbench/server.log')
        print(url)
        if args.open:webbrowser.open(url)
        return
    from workbench.server import make_server
    server=make_server(args.port)
    print(url,flush=True)
    if args.open:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.jobs.close();server.server_close()


if __name__=='__main__':main()
