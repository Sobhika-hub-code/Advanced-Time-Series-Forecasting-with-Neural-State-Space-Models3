import os
for root,dirs,files in os.walk('artifacts'):
    for f in files:
        print(os.path.join(root,f))
