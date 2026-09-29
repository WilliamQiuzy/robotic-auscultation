"""Serve only this review package on loopback, with video byte-range seeking."""
import argparse
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from sim.route_b_full_video_20260924.serve_review import ReviewHandler

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8772)
    args=parser.parse_args()
    ThreadingHTTPServer(('127.0.0.1',args.port),partial(ReviewHandler,
        directory=str(Path(__file__).resolve().parent))).serve_forever()
