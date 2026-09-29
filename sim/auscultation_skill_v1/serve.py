"""Serve a declared review bundle on 127.0.0.1:8773; no directory browsing."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from .config import ROOT
from .contract import read, require, sha


def resolve_request(directory, request, permitted):
    raw = urlsplit(request).path
    require(not re.search(r'%2f|%5c', raw, flags=re.I), 'encoded path separators are forbidden')
    decoded = unquote(raw)
    require(decoded.startswith('/') and '\\' not in decoded, 'invalid request path')
    relative = 'review.html' if decoded == '/' else decoded[1:]
    parts = relative.split('/')
    require(all(part and not part.startswith('.') for part in parts), 'dotfiles or traversal are forbidden')
    require(relative in permitted, 'file not in the review allowlist')
    directory = Path(directory).resolve(); path = directory / relative
    require(not any((directory.joinpath(*parts[:i])).is_symlink() for i in range(1, len(parts)+1)),
            'symlink serving is forbidden')
    path = path.resolve()
    require(directory in path.parents and path.is_file(), 'review file is unavailable')
    return path


def range_bounds(value, size):
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', value)
    require(match is not None and size > 0 and any(match.groups()), 'unsupported byte range')
    lo, hi = match.groups()
    if lo:
        start = int(lo); end = min(int(hi), size-1) if hi else size-1
    else:
        require(int(hi) > 0, 'empty suffix range')
        start = max(0, size-int(hi)); end = size-1
    require(0 <= start <= end < size, 'unsatisfiable byte range')
    return start, end


def verified_bundle(directory, package_root=ROOT):
    directory = Path(directory).resolve(); package_root = Path(package_root).resolve()
    require(package_root in directory.parents, 'server root must be a review bundle within this skill package')
    manifest = read(directory/'serve_manifest.json')
    require(manifest.get('schema') == 'auscultation_skill_review_bundle_v1', 'wrong review bundle schema')
    files = manifest.get('files', {})
    require({'review.html', 'summary.json'}.issubset(files), 'review bundle incomplete')
    allowed = {'review.html', 'summary.json', 'media/expert.mp4', 'media/learned.mp4'}
    require(set(files).issubset(allowed), 'undeclared review file type')
    fingerprints = {}
    for relative, record in files.items():
        path = resolve_request(directory, '/'+relative, files)
        require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], 'review bundle hash mismatch')
        stat = path.stat(); fingerprints[relative] = (stat.st_size, stat.st_mtime_ns)
    return files, fingerprints


class ReviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory, permitted, fingerprints, **kwargs):
        self.root = Path(directory).resolve(); self.permitted = permitted; self.fingerprints = fingerprints
        super().__init__(*args, directory=str(self.root), **kwargs)

    def end_headers(self):
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        super().end_headers()

    def send_head(self):
        self.remaining = 0
        try:
            path = resolve_request(self.root, self.path, self.permitted)
            relative = str(path.relative_to(self.root)); stat = path.stat()
            require((stat.st_size, stat.st_mtime_ns) == self.fingerprints[relative], 'bundle file changed since verification')
        except (OSError, ValueError, KeyError):
            self.send_error(404, 'Review file unavailable'); return None
        value = self.headers.get('Range')
        if value:
            try: start, end = range_bounds(value, stat.st_size)
            except ValueError:
                self.send_response(416); self.send_header('Content-Range', 'bytes */'+str(stat.st_size))
                self.send_header('Content-Length', '0'); self.end_headers(); return None
        else:
            start, end = 0, stat.st_size-1
        stream = path.open('rb'); stream.seek(start); self.remaining = end-start+1
        self.send_response(206 if value else 200)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(self.remaining))
        if value: self.send_header('Content-Range', f'bytes {start}-{end}/{stat.st_size}')
        self.end_headers(); return stream

    def copyfile(self, source, output):
        while self.remaining > 0:
            block = source.read(min(262144, self.remaining))
            if not block: break
            output.write(block); self.remaining -= len(block)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, default=ROOT/'results/review')
    parser.add_argument('--port', type=int, default=8773)
    args = parser.parse_args(argv)
    files, fingerprints = verified_bundle(args.review)
    handler = partial(ReviewHandler, directory=str(args.review), permitted=files, fingerprints=fingerprints)
    with ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
        print(f'Review: http://127.0.0.1:{args.port}/', flush=True)
        server.serve_forever()


if __name__ == '__main__': main()
