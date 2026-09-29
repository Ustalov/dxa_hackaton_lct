"""401 with the expected Basic challenge means the protected HTTP server is alive."""
import urllib.request
import urllib.error
try:
    with urllib.request.urlopen('http://127.0.0.1:8095/health',timeout=3) as response:
        assert response.status==200
except urllib.error.HTTPError as error:
    if error.code!=401 or 'ImageLab DXA' not in error.headers.get('WWW-Authenticate',''):
        raise
