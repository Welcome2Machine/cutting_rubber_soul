"""Segmentation, material tracking and sliding measurements from cutting videos."""
import importlib
import sys


COMMANDS = {
    'import': ('src.workflows.video_import', 'Compress and sample camera videos'),
    'track': ('src.workflows.tracking', 'Prepare and track fixed material identities'),
    'measure': ('src.workflows.measurement', 'Measure blade/contact geometry from saved tracks'),
    'motion': ('src.workflows.motion', 'Measure material motion relative to the blade'),
    'slip': ('src.workflows.sliding', 'Compute paired sliding estimates to explicit cutoffs'),
    'batch': ('src.workflows.batch', 'Run the complete pipeline from a case manifest'),
    'gallery': ('src.workflows.gallery', 'Render text-free reviews, static plots and PPT videos'),
    'optimize': ('src.workflows.optimize', 'Benchmark optional deformation and surface-mapping candidates'),
    'exchange-export': ('src.exchange.export', 'Export observations for finite-element comparison'),
    'exchange-check': ('src.exchange.validate', 'Validate a finite-element exchange package'),
    'check': ('src.workflows.check', 'Check retained local artifacts without computing'),
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ('-h', '--help'):
        print(__doc__ + '\n\nUsage: python softcut.py COMMAND [OPTIONS]\n')
        for name, (_, description) in COMMANDS.items():
            print(f'  {name:17s} {description}')
        print('\nUse COMMAND --help for its options.')
        return 0
    command = args.pop(0)
    if command not in COMMANDS:
        print(f'Unknown command: {command}. Use --help.', file=sys.stderr)
        return 2
    return importlib.import_module(COMMANDS[command][0]).main(args) or 0


if __name__ == '__main__':
    raise SystemExit(main())
