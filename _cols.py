import pandas as pd
from Add_excel_to_labelled import read_metadata_start_time, read_sensor_sheet_abs_times
xls = pd.ExcelFile(r"1029\\My Experiment 2025-10-29 13-57-39.xls")
start = read_metadata_start_time(xls, 'Metadata Time', 'START')
for name in xls.sheet_names:
    if name in ['Accelerometer','Gyroscope','Linear Acceleration','Gravity']:
        df = read_sensor_sheet_abs_times(xls, name, 'Time (s)', start)
        print(name, list(df.columns))
