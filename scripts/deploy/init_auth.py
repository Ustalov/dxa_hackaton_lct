import getpass,hashlib,json,os,secrets
from pathlib import Path
p=Path('.runtime/auth.json');p.parent.mkdir(parents=True,exist_ok=True)
if p.exists():
    print('Existing authorization kept:',p)
else:
    username=input('Login: ').strip()
    password=getpass.getpass('Password: ')
    if not username or ':' in username or len(password)<8:
        raise SystemExit('Use a nonempty login without : and a password of at least 8 characters')
    if password!=getpass.getpass('Repeat password: '):raise SystemExit('Passwords differ')
    salt=secrets.token_bytes(16);iterations=200000
    p.write_text(json.dumps(dict(username=username,salt=salt.hex(),iterations=iterations,
        password_hash=hashlib.pbkdf2_hmac('sha256',password.encode(),salt,iterations).hex())))
    p.chmod(0o600)
