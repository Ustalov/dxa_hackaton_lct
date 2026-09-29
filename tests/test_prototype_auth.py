import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from fastapi import FastAPI
from starlette.testclient import TestClient
from dxa.prototype_auth import PrototypeAuth


class AuthTests(unittest.TestCase):
    def test_all_routes_require_valid_credentials_even_after_success(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'auth.json';salt=b'test-only-salt'
            path.write_text(json.dumps(dict(username='test-user',salt=salt.hex(),iterations=1000,
                password_hash=hashlib.pbkdf2_hmac('sha256',b'test-password',salt,1000).hex())))
            app=FastAPI()
            app.add_middleware(PrototypeAuth,config_path=path)
            @app.get('/{path:path}')
            def read(path):return {'protected':True}
            with TestClient(app) as client:
                for route in ('/','/health','/docs','/api/jobs/id/results','/api/jobs/id/images/0.png'):
                    self.assertEqual(client.get(route).status_code,401)
                    self.assertEqual(client.get(route,auth=('test-user','incorrect')).status_code,401)
                    self.assertEqual(client.get(route,auth=('test-user','test-password')).status_code,200)
                    self.assertEqual(client.get(route).status_code,401)
                for header in ('Basic !!!','Bearer arbitrary','Basic '+base64.b64encode(b'no-colon').decode()):
                    response=client.get('/',headers={'Authorization':header})
                    self.assertEqual(response.status_code,401)
                    self.assertIn('Basic',response.headers['WWW-Authenticate'])


if __name__=='__main__':unittest.main()
