#!/usr/bin/env python

import json
import pandas as pd
import numpy as np
from glob import glob

for file_name in sorted(glob('./*/log.json')):
    print(file_name)
    file_name_new = file_name[:-5]+'.csv'
    with open(file_name,'r') as f:
        data = f.readlines()
    if len(data) == 0:
        print('\tWARNING: empty file')
        continue
    keys = json.loads(data[0]).keys()
    d = {k:[] for k in keys}
    for line in data:
        dat = json.loads(line)
        for k in keys:
            d[k].append(dat[k])
    data = {k:np.array(d[k]) for k in keys}
    df = pd.DataFrame.from_dict(data)
    df.to_csv(file_name_new, index=False)
