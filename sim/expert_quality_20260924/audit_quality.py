"""Read-only numerical expert quality audit; writes only the given new report."""
import argparse
import json
import sys

from .quality import audit_episode


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode')
    parser.add_argument('--output',required=True)
    parser.add_argument('--profile')
    args=parser.parse_args(argv)
    report=audit_episode(args.episode,args.output,args.profile)
    print(json.dumps({k:report[k] for k in ('overall_status','legacy_engineering_status','hard_failures','pending')},indent=2))
    return 1 if report['overall_status']=='rejected' else 0


if __name__=='__main__':
    sys.dont_write_bytecode=True
    raise SystemExit(main())
