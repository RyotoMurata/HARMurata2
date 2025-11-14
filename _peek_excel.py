import pandas as pd
path = r"1029\\My Experiment 2025-10-29 14-25-48.xls"
try:
    xls = pd.ExcelFile(path)
    print('sheets:', xls.sheet_names)
    for name in xls.sheet_names:
        try:
            df = xls.parse(name, nrows=8)
            print('---', name, '---')
            print(df.head().to_string())
        except Exception as e:
            print('Failed to parse', name, e)
except Exception as e:
    print('Excel open failed:', e)
