import os
print('YES' if os.getenv('GITHUB_TOKEN') else 'NO')
