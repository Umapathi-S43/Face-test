#!/usr/bin/env python3
"""
Simple HTTP server to serve the frontend during development.
In production, use nginx or serve from GitHub Pages / Vercel / etc.
"""

import os
import sys
import http.server
import argparse
from pathlib import Path
from functools import partial

FRONTEND_DIR = Path(__file__).parent


class CORSHandler(http.server.SimpleHTTPRequestHandler):
    """Simple handler with CORS headers."""

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description="Frontend dev server")
    parser.add_argument("--port", type=int, default=3000, help="Port (default: 3000)")
    parser.add_argument("--host", default="localhost", help="Host (default: localhost)")
    args = parser.parse_args()

    os.chdir(str(FRONTEND_DIR))
    handler = partial(CORSHandler, directory=str(FRONTEND_DIR))
    server = http.server.HTTPServer((args.host, args.port), handler)

    print(f"🌐 Frontend server: http://{args.host}:{args.port}")
    print(f"   Serving: {FRONTEND_DIR}")
    print(f"   Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹️ Stopped")
        server.server_close()


if __name__ == "__main__":
    main()
