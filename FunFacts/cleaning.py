import pandas as pd

#For both start and end time columns:
#Split end time into date and time, date col and time col
#Split date col into y/m/d cols

dirty = pd.read_csv('us_accidents_sample.csv')
dirty['Start_Time'].apply(lambda x: x.split(' ')[1])
dirty['Start_Time'].