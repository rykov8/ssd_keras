#!/usr/bin/env python

import json
import pandas as pd
from glob import glob

for file_name in sorted(glob('./*/history.json')):
    print(file_name)
    file_name_new = file_name[:-5]+'.csv'
    with open(file_name,'r') as f:
        data = f.read()
    data = json.loads(data)
    df = pd.DataFrame.from_dict(data)
    df.to_csv(file_name_new, index=False)
