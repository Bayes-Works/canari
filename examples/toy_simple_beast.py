import Rbeast as rb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# help(rb.load_example)
# help(rb.beast)

# Function to calculate fractional year
def fractional_year(dt):
    year_start = pd.Timestamp(year=dt.year, month=1, day=1)
    year_end = pd.Timestamp(year=dt.year + 1, month=1, day=1)
    year_length = (year_end - year_start).days
    elapsed_days = (dt - year_start).days
    return dt.year + elapsed_days / year_length


data_file = "./data/toy_time_series/synthetic_simple_autoregression_periodic.csv"
data = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None).squeeze()
datetime_file = "./data/toy_time_series/synthetic_simple_autoregression_periodic_datetime.csv"
year = pd.read_csv(datetime_file, skiprows=1, delimiter=",", header=None).squeeze()
year = pd.to_datetime(year)  # Convert to year objects
year = year.apply(fractional_year)  # Convert to fractional year

# LT anomaly
anm_mag = 0.5/52
anm_start_index = 52*10
anm_baseline = np.arange(len(data)) * anm_mag
anm_baseline[anm_start_index:] -= anm_baseline[anm_start_index]
anm_baseline[:anm_start_index] = 0
data = data.add(anm_baseline, axis=0)

# Remove the last 52*5 rows in df_raw
data = data[:-52*5]
year = year[:-52*5]

o = rb.beast(data, start=year.iloc[0], deltat=1/52, period=1.0)
# o = rb.beast(data, start=year.iloc[0], deltat=1/52, period=1.0, scp_minmax=[0,0], torder_minmax=[0,1])
# o = rb.beast(data, start=year.iloc[0], deltat=1/52, period=1.0, scp_minmax=[0,0], tcp_minmax=[0,5], torder_minmax=[0,1])

rb.plot(o, title='Beast - synthetic time series')
plt.show()
rb.print(o)


# nile, year = rb.load_example('nile')                     # nile is a 1d Python array or numpy vector
# o          = rb.beast( nile, start=1871, season='none')  # season='none' bcz the data has no seasonal/periodic component

# beach, year = rb.load_example('googletrend')
# o = rb.beast(beach, start= 2004, deltat=1/12, period = 1.0)       # the time unit is uknown or arbitrary
# o = rb.beast(beach, start= 2004, deltat=1/12, period ='1.0 year') # the time unit is fractional year

# rb.plot(o, title='Annual streamflow of the Nile River')
# plt.show()
# rb.print(o)

# Print a list of fields in the output variable (e.g, o.data, o.RMSE, o.trend.cp, o.time, and o.tend.cpOccPr)
# Check the R manual for expalanations of the output (https://cran.r-project.org/web/packages/Rbeast/Rbeast.pdf) 
# o                                                        # this is equivalent to "print(o)" 
