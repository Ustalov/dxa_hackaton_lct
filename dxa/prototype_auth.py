"""Password gate for all prototype routes, including images and API responses."""
import base64
import binascii
import hashlib
import hmac
import json
import os
from pathlib import Path
from starlette.responses import Response


class PrototypeAuth:
    def __init__(self, app, config_path):
        self.app = app
        config = json.loads(Path(config_path).read_text())
        self.username = config['username'].encode('utf-8')
        self.salt = bytes.fromhex(config['salt'])
        self.iterations = int(config['iterations'])
        self.password_hash = bytes.fromhex(config['password_hash'])
        self.cache_key = os.urandom(32)
        self.accepted_digest = None

    def valid(self, header):
        try:
            scheme, encoded = header.split(b' ', 1)
            if scheme.lower() != b'basic':
                return False
            credentials = base64.b64decode(encoded, validate=True)
            username, password = credentials.split(b':', 1)
            digest = hmac.digest(self.cache_key, credentials, 'sha256')
            if self.accepted_digest is not None and hmac.compare_digest(digest, self.accepted_digest):
                return True
            if not hmac.compare_digest(username, self.username):
                return False
            candidate = hashlib.pbkdf2_hmac('sha256', password, self.salt, self.iterations)
            if not hmac.compare_digest(candidate, self.password_hash):
                return False
            self.accepted_digest = digest
            return True
        except (ValueError, binascii.Error):
            return False

    async def __call__(self, scope, receive, send):
        if scope['type'] not in ('http', 'websocket'):
            return await self.app(scope, receive, send)
        header = dict(scope.get('headers', [])).get(b'authorization', b'')
        if self.valid(header):
            return await self.app(scope, receive, send)
        if scope['type'] == 'websocket':
            return await send({'type': 'websocket.close', 'code': 1008})
        response = Response('Authentication required', status_code=401, headers={
            'WWW-Authenticate': 'Basic realm="ImageLab DXA", charset="UTF-8"',
            'Cache-Control': 'no-store',
        })
        await response(scope, receive, send)
