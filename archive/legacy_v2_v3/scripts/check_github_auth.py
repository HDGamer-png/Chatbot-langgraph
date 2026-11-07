import os
import subprocess

print("GITHUB_TOKEN_SET=YES" if os.getenv('GITHUB_TOKEN') else "GITHUB_TOKEN_SET=NO")

try:
    out = subprocess.check_output(["gh","--version"], stderr=subprocess.STDOUT, text=True)
    print("GH_VERSION=" + out.splitlines()[0])
    try:
        auth = subprocess.check_output(["gh","auth","status","--hostname","github.com"], stderr=subprocess.STDOUT, text=True)
        print("GH_AUTH=OK")
    except subprocess.CalledProcessError as e:
        print("GH_AUTH=NOT_AUTH")
except FileNotFoundError:
    print("GH_VERSION=gh-not-found")
except Exception as e:
    print("GH_CHECK_ERROR=" + str(e))
