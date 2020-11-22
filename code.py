import sys
import pandas as pd
data = pd.read_json(sys.argv[1], lines=True)
data.to_csv('data.csv', index=False)