
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# === Begin: Pygrib_GFS Data Automated Download and Preprocessing_FINAL_U1000.ipynb ===

logger.info("!/usr/bin/env python")
#!/usr/bin/env python
logger.info("coding: utf-8")
# coding: utf-8

logger.info("In[1]:")
# In[1]:


import netCDF4


logger.info("In[2]:")
# In[2]:


from netCDF4 import Dataset


logger.info("In[3]:")
# In[3]:


import pandas as pd
import pygrib
import re
import numpy as np
from datetime import timedelta
import os


logger.info("In[4]:")
# In[4]:


cols= ['U1000']


logger.info("In[5]:")
# In[5]:


logger.info("Create a folder")
#Create a folder

logger.info("Base path")
# Base path
base_path = r"D:\D\Ruvision\GFS\Realtime GFS"

logger.info("Folder name from the list")
# Folder name from the list

folder_name = cols[0]

logger.info("Full path to create")
# Full path to create
folder_path = os.path.join(base_path, folder_name)

logger.info("Create the folder")
# Create the folder
os.makedirs(folder_path, exist_ok=True)

print(f"Folder created: {folder_path}")


logger.info("In[5]:")
# In[5]:


forecast_hr = np.arange(15, 85,3)


logger.info("In[6]:")
# In[6]:


forecast_hr


logger.info("In[7]:")
# In[7]:


from datetime import timedelta


logger.info("In[8]:")
# In[8]:


def find_key(dictionary, element):
    for key, value in dictionary.items():
        if element in value:
            return key
    return 'None'


logger.info("In[9]:")
# In[9]:


latbounds = [22.5 - 0.25, 23.5]
lonbounds = [72 , 73 + 0.25]


logger.info("In[10]:")
# In[10]:


time_from_ref = np.arange(15,85,3)


logger.info("In[11]:")
# In[11]:


variable_name = cols[0]


logger.info("In[12]:")
# In[12]:


from datetime import datetime

logger.info("Get the current UTC date and time")
# Get the current UTC date and time
current_utc_datetime = datetime.utcnow().date()
current_utc_datetime


logger.info("In[13]:")
# In[13]:


import os
import requests
from datetime import datetime, timedelta

base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

logger.info("Create a directory to store the downloaded files")
# Create a directory to store the downloaded files
output_directory = f"D:\D\Ruvision\GFS\Realtime GFS\{variable_name}"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

logger.info("Define the date for the forecast (in UTC)")
# Define the date for the forecast (in UTC)
current_date = datetime.utcnow()
logger.info("current_date=current_date - timedelta(days=1)")
#current_date=current_date - timedelta(days=1)

logger.info("List of initialization times (00, 06, 12, and 18 UTC)")
# List of initialization times (00, 06, 12, and 18 UTC)
init_times = [6]

for time in init_times:
logger.info("Loop through forecast hours from 3 to 72 in 3-hour intervals")
    # Loop through forecast hours from 3 to 72 in 3-hour intervals
    for forecast_hour in range(15, 85, 3):
logger.info("Construct the URL for the current forecast hour and initialization time")
        # Construct the URL for the current forecast hour and initialization time
        url = f"{base_url}?dir=%2Fgfs.{current_date.strftime('%Y%m%d')}%2F{time:02d}%2Fatmos&file=gfs.t{time:02d}z.pgrb2.0p25.f{forecast_hour:03d}&var_UGRD=on&lev_1000_mb=on&subregion=&toplat=47&leftlon=55&rightlon=105&bottomlat=0"

logger.info("Extract the filename from the URL")
        # Extract the filename from the URL
        filename = f"gfs.{current_date.strftime('%Y%m%d')}.t{time:02d}z.pgrb2.0p25.f{forecast_hour:03d}"
        filepath = os.path.join(output_directory, filename)

logger.info("Send a GET request to the URL")
        # Send a GET request to the URL
        response = requests.get(url)

logger.info("Check if the request was successful (HTTP status code 200)")
        # Check if the request was successful (HTTP status code 200)
        if response.status_code == 200:
logger.info("Save the downloaded file")
            # Save the downloaded file
            with open(filepath, "wb") as file:
                file.write(response.content)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download: {url}, HTTP status code: {response.status_code}")


logger.info("In[14]:")
# In[14]:


columns_prec = []

for i in cols:
    for time_steps in forecast_hr:
        for j in np.arange(23.5,22.25,-0.25):
            for k in np.arange(72.0,73.25,0.25):
                columns_prec.append(f'{i}_{j}_{k}_{time_steps:03d}')


logger.info("In[15]:")
# In[15]:


import pandas as pd

logger.info("Get today's date at 06:00 UTC")
# Get today's date at 06:00 UTC
ts_06utc = pd.Timestamp.utcnow().normalize() + pd.Timedelta(hours=6)
logger.info("ts_06utc = (pd.Timestamp.utcnow() - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=6)")
#ts_06utc = (pd.Timestamp.utcnow() - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=6)
logger.info("Create DataFrame with just one timestamp (06:00 UTC)")
# Create DataFrame with just one timestamp (06:00 UTC)
data_prec = pd.DataFrame(index=[ts_06utc], columns=columns_prec)

logger.info("Remove timezone info (if any)")
# Remove timezone info (if any)
data_prec.index = data_prec.index.tz_localize(None)


logger.info("In[16]:")
# In[16]:


data_prec


logger.info("In[17]:")
# In[17]:


root_directory = f"D:\D\Ruvision\GFS\Realtime GFS\{variable_name}"
logger.info("Regular expression pattern to match the filenames in the format 'gfs.t00z.pgrb2.0p25.f021'")
# Regular expression pattern to match the filenames in the format "gfs.t00z.pgrb2.0p25.f021"
filename_pattern = r'gfs\.\d{8}\.t\d{2}z\.pgrb2\.0p25\.f\d{3}$'

logger.info("Dictionary to store the filenames grouped by directory")
# Dictionary to store the filenames grouped by directory
directory_names = {}

logger.info("Function to check if the filename matches the expected pattern")
# Function to check if the filename matches the expected pattern
def is_grib2_file(filename):
    return re.match(filename_pattern, filename)

logger.info("Walk through the root directory and its subdirectories")
# Walk through the root directory and its subdirectories
for dirpath, dirnames, filenames in os.walk(root_directory):
    x = str(dirpath).split(os.sep)[-1]
    directory_names[x] = []

logger.info("Filter the filenames based on the expected pattern and store them in the dictionary")
    # Filter the filenames based on the expected pattern and store them in the dictionary
    for filename in filenames:
        if is_grib2_file(filename):
            directory_names[x].append(filename)

logger.info("Print the filenames of the GRIB2 files in each directory")
# Print the filenames of the GRIB2 files in each directory
for directory, filenames in directory_names.items():
    print(f"Directory: {directory}")
    for filename in filenames:
        print(filename)
    print()  # Add an empty line to separate directories


logger.info("In[19]:")
# In[19]:


logger.info("Open the GRIB file")
# Open the GRIB file
filename = f"D:\D\Ruvision\GFS\Realtime GFS\{variable_name}\gfs.20250722.t06z.pgrb2.0p25.f015"
grbs = pygrib.open(filename)

logger.info("Print information about each GRIB message (parameter)")
# Print information about each GRIB message (parameter)
for grb in grbs:
    print(f"Parameter Name: {grb.name}")
    print(f"Level: {grb.level}")
    print(f"Units: {grb.units}")
    print(f"Values: {grb.values}")
    print(f"Grid Shape: {grb.values.shape}")
    print("----------")

logger.info("Close the GRIB file")
# Close the GRIB file
grbs.close()


logger.info("In[20]:")
# In[20]:


import pygrib
import pandas as pd
import numpy as np
import os
from datetime import timedelta

counter = 0

for time_step in data_prec.index:
    year = time_step.year
    month = time_step.month
    day = time_step.day
    ref_time = time_step.hour

logger.info("Create 3-hourly time index for forecast horizon (from +15h to +84h)")
    # Create 3-hourly time index for forecast horizon (from +15h to +84h)
    date_temp = pd.date_range(start=time_step + timedelta(hours=15), end=time_step + timedelta(hours=84), freq='3h')
    col_temp = np.arange(0, 25)  # 5x5 grid flattened
    data_temp = pd.DataFrame(index=date_temp, columns=col_temp, dtype=float)

    for time_lag in time_from_ref:
        filename = f'gfs.{year}{month:02d}{day:02d}.t06z.pgrb2.0p25.f{time_lag:03d}'
        directory = find_key(directory_names, filename)  # Should return a folder or None

logger.info("Calculate the corresponding forecast datetime")
        # Calculate the corresponding forecast datetime
        forecast_time = time_step + timedelta(hours=int(time_lag))

        if directory is not None and directory != 'None':
            file_path = os.path.join(root_directory, filename)

            try:
                grbs = pygrib.open(file_path)
                grb = grbs.select(name=f"{grb.name}")[0]  # Select U wind component

                data = grb.values  # DO NOT subtract 273.15 — wind is in m/s, not Kelvin
                lats, lons = grb.latlons()

logger.info("Subset the region (ensure valid slicing)")
                # Subset the region (ensure valid slicing)
                latli = 2
                latui = 7
                lonli = 2
                lonui = 7

                grid = data[latli:latui, lonli:lonui][::-1]  # Flip to match orientation
                flat = np.ravel(grid)

                if flat.shape[0] == 25:
                    data_temp.loc[forecast_time] = flat
                else:
                    print(f"[WARN] Unexpected grid shape in {filename} — got {flat.shape[0]} instead of 25.")
                    data_temp.loc[forecast_time] = np.nan

                grbs.close()

            except Exception as e:
                print(f"[ERROR] Reading file {filename}: {e}")
                data_temp.loc[forecast_time] = np.nan

        else:
            print(f"[MISSING] File not found or directory not available: {filename}")
            data_temp.loc[forecast_time] = np.nan

logger.info("Flatten the entire 3-day 5x5 grid (25 * 24 = 600 values) and assign to data_tmp")
    # Flatten the entire 3-day 5x5 grid (25 * 24 = 600 values) and assign to data_tmp
    try:
        flat_data = np.ravel(data_temp.values.astype(float))
        if flat_data.shape[0] == 600:
            data_prec.loc[time_step, data_prec.columns[:600]] = flat_data
        else:
            print(f"[ERROR] Unexpected flattened shape: {flat_data.shape[0]} at {time_step}")
            data_prec.loc[time_step, data_prec.columns[:600]] = np.nan
    except Exception as e:
        print(f"[ERROR] Assigning flattened data at {time_step}: {e}")
        data_prec.loc[time_step, data_prec.columns[:600]] = np.nan

    counter += 1
    if counter % 50 == 0:
        print(f'Processed {counter} time steps.')


logger.info("In[21]:")
# In[21]:


data_prec.isnull().sum().max()


logger.info("In[22]:")
# In[22]:


data_prec


logger.info("In[23]:")
# In[23]:


data_final_interpolated1=data_prec


logger.info("In[24]:")
# In[24]:


data_final_interpolated1 = data_final_interpolated1.shift(freq=pd.Timedelta(hours=5, minutes=30))


logger.info("In[25]:")
# In[25]:


data_final_interpolated1 = data_final_interpolated1.rename_axis('DateTime')


logger.info("In[26]:")
# In[26]:


extracted_rows = data_final_interpolated1[data_final_interpolated1.index.time == pd.Timestamp("11:30").time()]


logger.info("In[27]:")
# In[27]:


data_prec_lead_day_1 = extracted_rows.loc[:, ~extracted_rows.columns.str.contains('|'.join(['039', '042', '045', '048', '051', '054', '057', '060', '063', '066', '069', '072', '075', '078', '081', '084']))]
data_prec_lead_day_2 = extracted_rows.loc[:, ~extracted_rows.columns.str.contains('|'.join(['015', '018', '021', '024', '027', '030', '033', '036', '063', '066', '069', '072', '075', '078', '081', '084']))]
data_prec_lead_day_3 = extracted_rows.loc[:, ~extracted_rows.columns.str.contains('|'.join(['015', '018', '021', '024', '027', '030', '033', '036', '039', '042', '045', '048', '051', '054', '057', '060']))]


logger.info("In[28]:")
# In[28]:


columns_prec = []
for i in cols:
    for j in np.arange(23.5,22.25,-0.25):
        for k in np.arange(72.0,73.25,0.25):
            columns_prec.append(f'{i}_{j}_{k}')


logger.info("In[29]:")
# In[29]:


start_date = extracted_rows.index[0]
end_date = extracted_rows.index[-1]


logger.info("In[30]:")
# In[30]:


logger.info("Update the start and end dates")
# Update the start and end dates
updated_start_date_1 = pd.to_datetime(start_date) + pd.Timedelta(hours=15)
updated_end_date_1 = pd.to_datetime(end_date) + pd.Timedelta(hours=36)
logger.info("Update the start and end dates")
# Update the start and end dates
updated_start_date_2 = pd.to_datetime(start_date) + pd.Timedelta(hours=39)
updated_end_date_2 = pd.to_datetime(end_date) + pd.Timedelta(hours=60)
logger.info("Update the start and end dates")
# Update the start and end dates
updated_start_date_3 = pd.to_datetime(start_date) + pd.Timedelta(hours=63)
updated_end_date_3 = pd.to_datetime(end_date) + pd.Timedelta(hours=84)

data_prec_1 = pd.DataFrame(index = pd.date_range(start=updated_start_date_1, end=updated_end_date_1, freq = '3h'), columns = columns_prec)
data_prec_2 = pd.DataFrame(index = pd.date_range(start=updated_start_date_2, end=updated_end_date_2, freq = '3h'), columns = columns_prec)
data_prec_3 = pd.DataFrame(index = pd.date_range(start=updated_start_date_3, end=updated_end_date_3, freq = '3h'), columns = columns_prec)


logger.info("In[31]:")
# In[31]:


selected_rows_1 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('02:30:00').time()]
selected_rows_2 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('05:30:00').time()]
selected_rows_3 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('08:30:00').time()]
selected_rows_4 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('11:30:00').time()]
selected_rows_5 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('14:30:00').time()]
selected_rows_6 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('17:30:00').time()]
selected_rows_7 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('20:30:00').time()]
selected_rows_8 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('23:30:00').time()]

selected_rows_9 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('02:30:00').time()]
selected_rows_10 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('05:30:00').time()]
selected_rows_11 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('08:30:00').time()]
selected_rows_12 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('11:30:00').time()]
selected_rows_13 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('14:30:00').time()]
selected_rows_14 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('17:30:00').time()]
selected_rows_15 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('20:30:00').time()]
selected_rows_16 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('23:30:00').time()]

selected_rows_17 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('02:30:00').time()]
selected_rows_18 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('05:30:00').time()]
selected_rows_19 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('08:30:00').time()]
selected_rows_20 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('11:30:00').time()]
selected_rows_21 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('14:30:00').time()]
selected_rows_22= data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('17:30:00').time()]
selected_rows_23 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('20:30:00').time()]
selected_rows_24 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('23:30:00').time()]


logger.info("In[32]:")
# In[32]:


x1=data_prec_lead_day_1.iloc[:, :25]
x2=data_prec_lead_day_1.iloc[:, 25:50]
x3=data_prec_lead_day_1.iloc[:, 50:75]
x4=data_prec_lead_day_1.iloc[:, 75:100]
x5=data_prec_lead_day_1.iloc[:, 100:125]
x6=data_prec_lead_day_1.iloc[:, 125:150]
x7=data_prec_lead_day_1.iloc[:, 150:175]
x8=data_prec_lead_day_1.iloc[:, 175:]

x9=data_prec_lead_day_2.iloc[:, :25]
x10=data_prec_lead_day_2.iloc[:, 25:50]
x11=data_prec_lead_day_2.iloc[:, 50:75]
x12=data_prec_lead_day_2.iloc[:, 75:100]
x13=data_prec_lead_day_2.iloc[:, 100:125]
x14=data_prec_lead_day_2.iloc[:, 125:150]
x15=data_prec_lead_day_2.iloc[:, 150:175]
x16=data_prec_lead_day_2.iloc[:, 175:]

x17=data_prec_lead_day_3.iloc[:, :25]
x18=data_prec_lead_day_3.iloc[:, 25:50]
x19=data_prec_lead_day_3.iloc[:, 50:75]
x20=data_prec_lead_day_3.iloc[:, 75:100]
x21=data_prec_lead_day_3.iloc[:, 100:125]
x22=data_prec_lead_day_3.iloc[:, 125:150]
x23=data_prec_lead_day_3.iloc[:, 150:175]
x24=data_prec_lead_day_3.iloc[:, 175:]


logger.info("In[33]:")
# In[33]:


selected_rows_1.loc[:, :] = x1.values
selected_rows_2.loc[:, :] = x2.values
selected_rows_3.loc[:, :] = x3.values
selected_rows_4.loc[:, :] = x4.values
selected_rows_5.loc[:, :] = x5.values
selected_rows_6.loc[:, :] = x6.values
selected_rows_7.loc[:, :] = x7.values
selected_rows_8.loc[:, :] = x8.values
selected_rows_9.loc[:, :] = x9.values
selected_rows_10.loc[:, :] = x10.values
selected_rows_11.loc[:, :] = x11.values
selected_rows_12.loc[:, :] = x12.values
selected_rows_13.loc[:, :] = x13.values
selected_rows_14.loc[:, :] = x14.values
selected_rows_15.loc[:, :] = x15.values
selected_rows_16.loc[:, :] = x16.values
selected_rows_17.loc[:, :] = x17.values
selected_rows_18.loc[:, :] = x18.values
selected_rows_19.loc[:, :] = x19.values
selected_rows_20.loc[:, :] = x20.values
selected_rows_21.loc[:, :] = x21.values
selected_rows_22.loc[:, :] = x22.values
selected_rows_23.loc[:, :] = x23.values
selected_rows_24.loc[:, :] = x24.values


logger.info("In[34]:")
# In[34]:


merged_df_1 = pd.concat([selected_rows_1, selected_rows_2, selected_rows_3, selected_rows_4, 
                       selected_rows_5, selected_rows_6, selected_rows_7, selected_rows_8], axis=0)

merged_df_2 = pd.concat([selected_rows_9, selected_rows_10, selected_rows_11, selected_rows_12, 
                       selected_rows_13, selected_rows_14, selected_rows_15, selected_rows_16], axis=0)

merged_df_3 = pd.concat([selected_rows_17, selected_rows_18, selected_rows_19, selected_rows_20, 
                       selected_rows_21, selected_rows_22, selected_rows_23, selected_rows_24], axis=0)

merged_df_1 = merged_df_1.rename_axis('DateTime')
merged_df_1.reset_index('DateTime', inplace=True)
sorted_df_1 = merged_df_1.sort_values(by='DateTime', ascending=True)
sorted_df_1.set_index('DateTime', inplace=True)
data_X_Lead_Day_1=sorted_df_1


merged_df_2 = merged_df_2.rename_axis('DateTime')
merged_df_2.reset_index('DateTime', inplace=True)
sorted_df_2 = merged_df_2.sort_values(by='DateTime', ascending=True)
sorted_df_2.set_index('DateTime', inplace=True)
data_X_Lead_Day_2=sorted_df_2

merged_df_3 = merged_df_3.rename_axis('DateTime')
merged_df_3.reset_index('DateTime', inplace=True)
sorted_df_3 = merged_df_3.sort_values(by='DateTime', ascending=True)
sorted_df_3.set_index('DateTime', inplace=True)
data_X_Lead_Day_3=sorted_df_3

group_idx_1 = (data_X_Lead_Day_1.index.to_series().reset_index(drop=True).index // 8)
group_idx_2 = (data_X_Lead_Day_2.index.to_series().reset_index(drop=True).index // 8)
group_idx_3 = (data_X_Lead_Day_3.index.to_series().reset_index(drop=True).index // 8)

summed_data_1 = data_X_Lead_Day_1.groupby(group_idx_1).mean()
summed_data_2 = data_X_Lead_Day_2.groupby(group_idx_2).mean()
summed_data_3 = data_X_Lead_Day_3.groupby(group_idx_3).mean()

logger.info("Get today's date normalized to midnight")
# Get today's date normalized to midnight
today = pd.Timestamp.today().normalize()
logger.info("today = pd.Timestamp.today().normalize()- pd.Timedelta(days=1)")
#today = pd.Timestamp.today().normalize()- pd.Timedelta(days=1)

logger.info("Generate dates for tomorrow, day after, and two days after — all at 23:30")
# Generate dates for tomorrow, day after, and two days after — all at 23:30
date1 = today + pd.Timedelta(days=1, hours=23, minutes=30)
date2 = today + pd.Timedelta(days=2, hours=23, minutes=30)
date3 = today + pd.Timedelta(days=3, hours=23, minutes=30)

logger.info("Create DataFrames")
# Create DataFrames
data_prec_1 = pd.DataFrame(index=pd.date_range(start=date1, end=date1, freq='24h'), columns=summed_data_1.columns)
data_prec_2 = pd.DataFrame(index=pd.date_range(start=date2, end=date2, freq='24h'), columns=summed_data_2.columns)
data_prec_3 = pd.DataFrame(index=pd.date_range(start=date3, end=date3, freq='24h'), columns=summed_data_3.columns)

summed_data_1['DateTime']= data_prec_1.index
summed_data_2['DateTime']= data_prec_2.index
summed_data_3['DateTime']= data_prec_3.index

summed_data_1.set_index('DateTime', inplace=True)
summed_data_2.set_index('DateTime', inplace=True)
summed_data_3.set_index('DateTime', inplace=True)


logger.info("In[35]:")
# In[35]:


summed_data_1


logger.info("In[36]:")
# In[36]:


X1 = pd.read_excel(f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 1_daily basis.xlsx")
X2 = pd.read_excel(f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 2_daily basis.xlsx")
X3 = pd.read_excel(f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 3_daily basis.xlsx")


logger.info("In[37]:")
# In[37]:


X1.set_index('DateTime', inplace=True)
X2.set_index('DateTime', inplace=True)
X3.set_index('DateTime', inplace=True)


logger.info("In[38]:")
# In[38]:


Data_X1= pd.concat([X1, summed_data_1], axis=0)
Data_X2= pd.concat([X2, summed_data_2], axis=0)
Data_X3= pd.concat([X3, summed_data_3], axis=0)


logger.info("In[39]:")
# In[39]:


Data_X1


logger.info("In[40]:")
# In[40]:


filename_1 = f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 1_daily basis.xlsx"
filename_2 = f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 2_daily basis.xlsx"
filename_3 = f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 3_daily basis.xlsx"

logger.info("Save the DataFrame")
# Save the DataFrame
Data_X1.to_excel(filename_1)
Data_X2.to_excel(filename_2)
Data_X3.to_excel(filename_3)


logger.info("In[ ]:")
# In[ ]:





logger.info("In[ ]:")
# In[ ]:





logger.info("In[ ]:")
# In[ ]:
# === End: Pygrib_GFS Data Automated Download and Preprocessing_FINAL_U1000.ipynb ===


# === Begin: Pygrib_GFS Data Automated Download and Preprocessing_FINAL_V1000.ipynb ===

logger.info("!/usr/bin/env python")
#!/usr/bin/env python
logger.info("coding: utf-8")
# coding: utf-8

logger.info("In[1]:")
# In[1]:


import netCDF4


logger.info("In[2]:")
# In[2]:


from netCDF4 import Dataset


logger.info("In[3]:")
# In[3]:


import pandas as pd
import pygrib
import re
import numpy as np
from datetime import timedelta
import os


logger.info("In[4]:")
# In[4]:


cols= ['V1000']


logger.info("In[5]:")
# In[5]:


logger.info("Create a folder")
#Create a folder

logger.info("Base path")
# Base path
base_path = r"D:\D\Ruvision\GFS\Realtime GFS"

logger.info("Folder name from the list")
# Folder name from the list

folder_name = cols[0]

logger.info("Full path to create")
# Full path to create
folder_path = os.path.join(base_path, folder_name)

logger.info("Create the folder")
# Create the folder
os.makedirs(folder_path, exist_ok=True)

print(f"Folder created: {folder_path}")


logger.info("In[6]:")
# In[6]:


forecast_hr = np.arange(15, 85,3)


logger.info("In[7]:")
# In[7]:


forecast_hr


logger.info("In[8]:")
# In[8]:


from datetime import timedelta


logger.info("In[9]:")
# In[9]:


def find_key(dictionary, element):
    for key, value in dictionary.items():
        if element in value:
            return key
    return 'None'


logger.info("In[10]:")
# In[10]:


latbounds = [22.5 - 0.25, 23.5]
lonbounds = [72 , 73 + 0.25]


logger.info("In[11]:")
# In[11]:


time_from_ref = np.arange(15,85,3)


logger.info("In[12]:")
# In[12]:


variable_name = cols[0]


logger.info("In[13]:")
# In[13]:


from datetime import datetime

logger.info("Get the current UTC date and time")
# Get the current UTC date and time
current_utc_datetime = datetime.utcnow().date()
current_utc_datetime


logger.info("In[15]:")
# In[15]:


import os
import requests
from datetime import datetime, timedelta

base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

logger.info("Create a directory to store the downloaded files")
# Create a directory to store the downloaded files
output_directory = f"D:\D\Ruvision\GFS\Realtime GFS\{variable_name}"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

logger.info("Define the date for the forecast (in UTC)")
# Define the date for the forecast (in UTC)
current_date = datetime.utcnow()
logger.info("current_date=current_date - timedelta(days=1)")
#current_date=current_date - timedelta(days=1)

logger.info("List of initialization times (00, 06, 12, and 18 UTC)")
# List of initialization times (00, 06, 12, and 18 UTC)
init_times = [6]

for time in init_times:
logger.info("Loop through forecast hours from 3 to 72 in 3-hour intervals")
    # Loop through forecast hours from 3 to 72 in 3-hour intervals
    for forecast_hour in range(15, 85, 3):
logger.info("Construct the URL for the current forecast hour and initialization time")
        # Construct the URL for the current forecast hour and initialization time
        url = f"{base_url}?dir=%2Fgfs.{current_date.strftime('%Y%m%d')}%2F{time:02d}%2Fatmos&file=gfs.t{time:02d}z.pgrb2.0p25.f{forecast_hour:03d}&var_VGRD=on&lev_1000_mb=on&subregion=&toplat=47&leftlon=55&rightlon=105&bottomlat=0"

logger.info("Extract the filename from the URL")
        # Extract the filename from the URL
        filename = f"gfs.{current_date.strftime('%Y%m%d')}.t{time:02d}z.pgrb2.0p25.f{forecast_hour:03d}"
        filepath = os.path.join(output_directory, filename)

logger.info("Send a GET request to the URL")
        # Send a GET request to the URL
        response = requests.get(url)

logger.info("Check if the request was successful (HTTP status code 200)")
        # Check if the request was successful (HTTP status code 200)
        if response.status_code == 200:
logger.info("Save the downloaded file")
            # Save the downloaded file
            with open(filepath, "wb") as file:
                file.write(response.content)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download: {url}, HTTP status code: {response.status_code}")


logger.info("In[16]:")
# In[16]:


columns_prec = []

for i in cols:
    for time_steps in forecast_hr:
        for j in np.arange(23.5,22.25,-0.25):
            for k in np.arange(72.0,73.25,0.25):
                columns_prec.append(f'{i}_{j}_{k}_{time_steps:03d}')


logger.info("In[17]:")
# In[17]:


import pandas as pd

logger.info("Get today's date at 06:00 UTC")
# Get today's date at 06:00 UTC
ts_06utc = pd.Timestamp.utcnow().normalize() + pd.Timedelta(hours=6)
logger.info("ts_06utc = (pd.Timestamp.utcnow() - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=6)")
#ts_06utc = (pd.Timestamp.utcnow() - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=6)

logger.info("Create DataFrame with just one timestamp (06:00 UTC)")
# Create DataFrame with just one timestamp (06:00 UTC)
data_prec = pd.DataFrame(index=[ts_06utc], columns=columns_prec)

logger.info("Remove timezone info (if any)")
# Remove timezone info (if any)
data_prec.index = data_prec.index.tz_localize(None)


logger.info("In[18]:")
# In[18]:


data_prec


logger.info("In[19]:")
# In[19]:


root_directory = f"D:\D\Ruvision\GFS\Realtime GFS\{variable_name}"
logger.info("Regular expression pattern to match the filenames in the format 'gfs.t00z.pgrb2.0p25.f021'")
# Regular expression pattern to match the filenames in the format "gfs.t00z.pgrb2.0p25.f021"
filename_pattern = r'gfs\.\d{8}\.t\d{2}z\.pgrb2\.0p25\.f\d{3}$'

logger.info("Dictionary to store the filenames grouped by directory")
# Dictionary to store the filenames grouped by directory
directory_names = {}

logger.info("Function to check if the filename matches the expected pattern")
# Function to check if the filename matches the expected pattern
def is_grib2_file(filename):
    return re.match(filename_pattern, filename)

logger.info("Walk through the root directory and its subdirectories")
# Walk through the root directory and its subdirectories
for dirpath, dirnames, filenames in os.walk(root_directory):
    x = str(dirpath).split(os.sep)[-1]
    directory_names[x] = []

logger.info("Filter the filenames based on the expected pattern and store them in the dictionary")
    # Filter the filenames based on the expected pattern and store them in the dictionary
    for filename in filenames:
        if is_grib2_file(filename):
            directory_names[x].append(filename)

logger.info("Print the filenames of the GRIB2 files in each directory")
# Print the filenames of the GRIB2 files in each directory
for directory, filenames in directory_names.items():
    print(f"Directory: {directory}")
    for filename in filenames:
        print(filename)
    print()  # Add an empty line to separate directories


logger.info("In[20]:")
# In[20]:


logger.info("Open the GRIB file")
# Open the GRIB file
filename = f"D:\D\Ruvision\GFS\Realtime GFS\{variable_name}\gfs.20250722.t06z.pgrb2.0p25.f015"
grbs = pygrib.open(filename)

logger.info("Print information about each GRIB message (parameter)")
# Print information about each GRIB message (parameter)
for grb in grbs:
    print(f"Parameter Name: {grb.name}")
    print(f"Level: {grb.level}")
    print(f"Units: {grb.units}")
    print(f"Values: {grb.values}")
    print(f"Grid Shape: {grb.values.shape}")
    print("----------")

logger.info("Close the GRIB file")
# Close the GRIB file
grbs.close()


logger.info("In[21]:")
# In[21]:


import pygrib
import pandas as pd
import numpy as np
import os
from datetime import timedelta

counter = 0

for time_step in data_prec.index:
    year = time_step.year
    month = time_step.month
    day = time_step.day
    ref_time = time_step.hour

logger.info("Create 3-hourly time index for forecast horizon (from +15h to +84h)")
    # Create 3-hourly time index for forecast horizon (from +15h to +84h)
    date_temp = pd.date_range(start=time_step + timedelta(hours=15), end=time_step + timedelta(hours=84), freq='3h')
    col_temp = np.arange(0, 25)  # 5x5 grid flattened
    data_temp = pd.DataFrame(index=date_temp, columns=col_temp, dtype=float)

    for time_lag in time_from_ref:
        filename = f'gfs.{year}{month:02d}{day:02d}.t06z.pgrb2.0p25.f{time_lag:03d}'
        directory = find_key(directory_names, filename)  # Should return a folder or None

logger.info("Calculate the corresponding forecast datetime")
        # Calculate the corresponding forecast datetime
        forecast_time = time_step + timedelta(hours=int(time_lag))

        if directory is not None and directory != 'None':
            file_path = os.path.join(root_directory, filename)

            try:
                grbs = pygrib.open(file_path)
                grb = grbs.select(name=f"{grb.name}")[0]  # Select U wind component

                data = grb.values  # DO NOT subtract 273.15 — wind is in m/s, not Kelvin
                lats, lons = grb.latlons()

logger.info("Subset the region (ensure valid slicing)")
                # Subset the region (ensure valid slicing)
                latli = 2
                latui = 7
                lonli = 2
                lonui = 7

                grid = data[latli:latui, lonli:lonui][::-1]  # Flip to match orientation
                flat = np.ravel(grid)

                if flat.shape[0] == 25:
                    data_temp.loc[forecast_time] = flat
                else:
                    print(f"[WARN] Unexpected grid shape in {filename} — got {flat.shape[0]} instead of 25.")
                    data_temp.loc[forecast_time] = np.nan

                grbs.close()

            except Exception as e:
                print(f"[ERROR] Reading file {filename}: {e}")
                data_temp.loc[forecast_time] = np.nan

        else:
            print(f"[MISSING] File not found or directory not available: {filename}")
            data_temp.loc[forecast_time] = np.nan

logger.info("Flatten the entire 3-day 5x5 grid (25 * 24 = 600 values) and assign to data_tmp")
    # Flatten the entire 3-day 5x5 grid (25 * 24 = 600 values) and assign to data_tmp
    try:
        flat_data = np.ravel(data_temp.values.astype(float))
        if flat_data.shape[0] == 600:
            data_prec.loc[time_step, data_prec.columns[:600]] = flat_data
        else:
            print(f"[ERROR] Unexpected flattened shape: {flat_data.shape[0]} at {time_step}")
            data_prec.loc[time_step, data_prec.columns[:600]] = np.nan
    except Exception as e:
        print(f"[ERROR] Assigning flattened data at {time_step}: {e}")
        data_prec.loc[time_step, data_prec.columns[:600]] = np.nan

    counter += 1
    if counter % 50 == 0:
        print(f'Processed {counter} time steps.')


logger.info("In[22]:")
# In[22]:


data_prec.isnull().sum().max()


logger.info("In[23]:")
# In[23]:


data_prec


logger.info("In[24]:")
# In[24]:


data_final_interpolated1=data_prec


logger.info("In[25]:")
# In[25]:


data_final_interpolated1 = data_final_interpolated1.shift(freq=pd.Timedelta(hours=5, minutes=30))


logger.info("In[26]:")
# In[26]:


data_final_interpolated1 = data_final_interpolated1.rename_axis('DateTime')


logger.info("In[27]:")
# In[27]:


extracted_rows = data_final_interpolated1[data_final_interpolated1.index.time == pd.Timestamp("11:30").time()]


logger.info("In[28]:")
# In[28]:


data_prec_lead_day_1 = extracted_rows.loc[:, ~extracted_rows.columns.str.contains('|'.join(['039', '042', '045', '048', '051', '054', '057', '060', '063', '066', '069', '072', '075', '078', '081', '084']))]
data_prec_lead_day_2 = extracted_rows.loc[:, ~extracted_rows.columns.str.contains('|'.join(['015', '018', '021', '024', '027', '030', '033', '036', '063', '066', '069', '072', '075', '078', '081', '084']))]
data_prec_lead_day_3 = extracted_rows.loc[:, ~extracted_rows.columns.str.contains('|'.join(['015', '018', '021', '024', '027', '030', '033', '036', '039', '042', '045', '048', '051', '054', '057', '060']))]


logger.info("In[29]:")
# In[29]:


columns_prec = []
for i in cols:
    for j in np.arange(23.5,22.25,-0.25):
        for k in np.arange(72.0,73.25,0.25):
            columns_prec.append(f'{i}_{j}_{k}')


logger.info("In[30]:")
# In[30]:


start_date = extracted_rows.index[0]
end_date = extracted_rows.index[-1]


logger.info("In[31]:")
# In[31]:


logger.info("Update the start and end dates")
# Update the start and end dates
updated_start_date_1 = pd.to_datetime(start_date) + pd.Timedelta(hours=15)
updated_end_date_1 = pd.to_datetime(end_date) + pd.Timedelta(hours=36)
logger.info("Update the start and end dates")
# Update the start and end dates
updated_start_date_2 = pd.to_datetime(start_date) + pd.Timedelta(hours=39)
updated_end_date_2 = pd.to_datetime(end_date) + pd.Timedelta(hours=60)
logger.info("Update the start and end dates")
# Update the start and end dates
updated_start_date_3 = pd.to_datetime(start_date) + pd.Timedelta(hours=63)
updated_end_date_3 = pd.to_datetime(end_date) + pd.Timedelta(hours=84)

data_prec_1 = pd.DataFrame(index = pd.date_range(start=updated_start_date_1, end=updated_end_date_1, freq = '3h'), columns = columns_prec)
data_prec_2 = pd.DataFrame(index = pd.date_range(start=updated_start_date_2, end=updated_end_date_2, freq = '3h'), columns = columns_prec)
data_prec_3 = pd.DataFrame(index = pd.date_range(start=updated_start_date_3, end=updated_end_date_3, freq = '3h'), columns = columns_prec)


logger.info("In[32]:")
# In[32]:


selected_rows_1 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('02:30:00').time()]
selected_rows_2 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('05:30:00').time()]
selected_rows_3 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('08:30:00').time()]
selected_rows_4 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('11:30:00').time()]
selected_rows_5 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('14:30:00').time()]
selected_rows_6 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('17:30:00').time()]
selected_rows_7 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('20:30:00').time()]
selected_rows_8 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('23:30:00').time()]

selected_rows_9 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('02:30:00').time()]
selected_rows_10 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('05:30:00').time()]
selected_rows_11 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('08:30:00').time()]
selected_rows_12 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('11:30:00').time()]
selected_rows_13 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('14:30:00').time()]
selected_rows_14 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('17:30:00').time()]
selected_rows_15 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('20:30:00').time()]
selected_rows_16 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('23:30:00').time()]

selected_rows_17 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('02:30:00').time()]
selected_rows_18 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('05:30:00').time()]
selected_rows_19 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('08:30:00').time()]
selected_rows_20 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('11:30:00').time()]
selected_rows_21 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('14:30:00').time()]
selected_rows_22= data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('17:30:00').time()]
selected_rows_23 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('20:30:00').time()]
selected_rows_24 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('23:30:00').time()]


logger.info("In[33]:")
# In[33]:


x1=data_prec_lead_day_1.iloc[:, :25]
x2=data_prec_lead_day_1.iloc[:, 25:50]
x3=data_prec_lead_day_1.iloc[:, 50:75]
x4=data_prec_lead_day_1.iloc[:, 75:100]
x5=data_prec_lead_day_1.iloc[:, 100:125]
x6=data_prec_lead_day_1.iloc[:, 125:150]
x7=data_prec_lead_day_1.iloc[:, 150:175]
x8=data_prec_lead_day_1.iloc[:, 175:]

x9=data_prec_lead_day_2.iloc[:, :25]
x10=data_prec_lead_day_2.iloc[:, 25:50]
x11=data_prec_lead_day_2.iloc[:, 50:75]
x12=data_prec_lead_day_2.iloc[:, 75:100]
x13=data_prec_lead_day_2.iloc[:, 100:125]
x14=data_prec_lead_day_2.iloc[:, 125:150]
x15=data_prec_lead_day_2.iloc[:, 150:175]
x16=data_prec_lead_day_2.iloc[:, 175:]

x17=data_prec_lead_day_3.iloc[:, :25]
x18=data_prec_lead_day_3.iloc[:, 25:50]
x19=data_prec_lead_day_3.iloc[:, 50:75]
x20=data_prec_lead_day_3.iloc[:, 75:100]
x21=data_prec_lead_day_3.iloc[:, 100:125]
x22=data_prec_lead_day_3.iloc[:, 125:150]
x23=data_prec_lead_day_3.iloc[:, 150:175]
x24=data_prec_lead_day_3.iloc[:, 175:]


logger.info("In[34]:")
# In[34]:


selected_rows_1.loc[:, :] = x1.values
selected_rows_2.loc[:, :] = x2.values
selected_rows_3.loc[:, :] = x3.values
selected_rows_4.loc[:, :] = x4.values
selected_rows_5.loc[:, :] = x5.values
selected_rows_6.loc[:, :] = x6.values
selected_rows_7.loc[:, :] = x7.values
selected_rows_8.loc[:, :] = x8.values
selected_rows_9.loc[:, :] = x9.values
selected_rows_10.loc[:, :] = x10.values
selected_rows_11.loc[:, :] = x11.values
selected_rows_12.loc[:, :] = x12.values
selected_rows_13.loc[:, :] = x13.values
selected_rows_14.loc[:, :] = x14.values
selected_rows_15.loc[:, :] = x15.values
selected_rows_16.loc[:, :] = x16.values
selected_rows_17.loc[:, :] = x17.values
selected_rows_18.loc[:, :] = x18.values
selected_rows_19.loc[:, :] = x19.values
selected_rows_20.loc[:, :] = x20.values
selected_rows_21.loc[:, :] = x21.values
selected_rows_22.loc[:, :] = x22.values
selected_rows_23.loc[:, :] = x23.values
selected_rows_24.loc[:, :] = x24.values


logger.info("In[35]:")
# In[35]:


merged_df_1 = pd.concat([selected_rows_1, selected_rows_2, selected_rows_3, selected_rows_4, 
                       selected_rows_5, selected_rows_6, selected_rows_7, selected_rows_8], axis=0)

merged_df_2 = pd.concat([selected_rows_9, selected_rows_10, selected_rows_11, selected_rows_12, 
                       selected_rows_13, selected_rows_14, selected_rows_15, selected_rows_16], axis=0)

merged_df_3 = pd.concat([selected_rows_17, selected_rows_18, selected_rows_19, selected_rows_20, 
                       selected_rows_21, selected_rows_22, selected_rows_23, selected_rows_24], axis=0)

merged_df_1 = merged_df_1.rename_axis('DateTime')
merged_df_1.reset_index('DateTime', inplace=True)
sorted_df_1 = merged_df_1.sort_values(by='DateTime', ascending=True)
sorted_df_1.set_index('DateTime', inplace=True)
data_X_Lead_Day_1=sorted_df_1


merged_df_2 = merged_df_2.rename_axis('DateTime')
merged_df_2.reset_index('DateTime', inplace=True)
sorted_df_2 = merged_df_2.sort_values(by='DateTime', ascending=True)
sorted_df_2.set_index('DateTime', inplace=True)
data_X_Lead_Day_2=sorted_df_2

merged_df_3 = merged_df_3.rename_axis('DateTime')
merged_df_3.reset_index('DateTime', inplace=True)
sorted_df_3 = merged_df_3.sort_values(by='DateTime', ascending=True)
sorted_df_3.set_index('DateTime', inplace=True)
data_X_Lead_Day_3=sorted_df_3

group_idx_1 = (data_X_Lead_Day_1.index.to_series().reset_index(drop=True).index // 8)
group_idx_2 = (data_X_Lead_Day_2.index.to_series().reset_index(drop=True).index // 8)
group_idx_3 = (data_X_Lead_Day_3.index.to_series().reset_index(drop=True).index // 8)

summed_data_1 = data_X_Lead_Day_1.groupby(group_idx_1).mean()
summed_data_2 = data_X_Lead_Day_2.groupby(group_idx_2).mean()
summed_data_3 = data_X_Lead_Day_3.groupby(group_idx_3).mean()

logger.info("Get today's date normalized to midnight")
# Get today's date normalized to midnight
today = pd.Timestamp.today().normalize()
logger.info("today = pd.Timestamp.today().normalize()- pd.Timedelta(days=1)")
#today = pd.Timestamp.today().normalize()- pd.Timedelta(days=1)

logger.info("Generate dates for tomorrow, day after, and two days after — all at 23:30")
# Generate dates for tomorrow, day after, and two days after — all at 23:30
date1 = today + pd.Timedelta(days=1, hours=23, minutes=30)
date2 = today + pd.Timedelta(days=2, hours=23, minutes=30)
date3 = today + pd.Timedelta(days=3, hours=23, minutes=30)

logger.info("Create DataFrames")
# Create DataFrames
data_prec_1 = pd.DataFrame(index=pd.date_range(start=date1, end=date1, freq='24h'), columns=summed_data_1.columns)
data_prec_2 = pd.DataFrame(index=pd.date_range(start=date2, end=date2, freq='24h'), columns=summed_data_2.columns)
data_prec_3 = pd.DataFrame(index=pd.date_range(start=date3, end=date3, freq='24h'), columns=summed_data_3.columns)

summed_data_1['DateTime']= data_prec_1.index
summed_data_2['DateTime']= data_prec_2.index
summed_data_3['DateTime']= data_prec_3.index

summed_data_1.set_index('DateTime', inplace=True)
summed_data_2.set_index('DateTime', inplace=True)
summed_data_3.set_index('DateTime', inplace=True)


logger.info("In[38]:")
# In[38]:


summed_data_1


logger.info("In[39]:")
# In[39]:


X1 = pd.read_excel(f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 1_daily basis.xlsx")
X2 = pd.read_excel(f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 2_daily basis.xlsx")
X3 = pd.read_excel(f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 3_daily basis.xlsx")


logger.info("In[40]:")
# In[40]:


X1.set_index('DateTime', inplace=True)
X2.set_index('DateTime', inplace=True)
X3.set_index('DateTime', inplace=True)


logger.info("In[41]:")
# In[41]:


Data_X1= pd.concat([X1, summed_data_1], axis=0)
Data_X2= pd.concat([X2, summed_data_2], axis=0)
Data_X3= pd.concat([X3, summed_data_3], axis=0)


logger.info("In[42]:")
# In[42]:


Data_X2


logger.info("In[44]:")
# In[44]:


filename_1 = f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 1_daily basis.xlsx"
filename_2 = f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 2_daily basis.xlsx"
filename_3 = f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 3_daily basis.xlsx"

logger.info("Save the DataFrame")
# Save the DataFrame
Data_X1.to_excel(filename_1)
Data_X2.to_excel(filename_2)
Data_X3.to_excel(filename_3)


logger.info("In[ ]:")
# In[ ]:





logger.info("In[ ]:")
# In[ ]:





logger.info("In[ ]:")
# In[ ]:
# === End: Pygrib_GFS Data Automated Download and Preprocessing_FINAL_V1000.ipynb ===


# === Begin: Pygrib_GFS Rain Data Automated Download and Preprocessing_RUVISION_FINAL_22-07-2025.ipynb ===

logger.info("!/usr/bin/env python")
#!/usr/bin/env python
logger.info("coding: utf-8")
# coding: utf-8

logger.info("In[1]:")
# In[1]:


import netCDF4


logger.info("In[2]:")
# In[2]:


from netCDF4 import Dataset


logger.info("In[3]:")
# In[3]:


import pandas as pd
import pygrib
import re
import numpy as np
from datetime import timedelta
import os


logger.info("In[4]:")
# In[4]:


cols= ['PREC']


logger.info("In[5]:")
# In[5]:


logger.info("Create a folder")
#Create a folder

logger.info("Base path")
# Base path
base_path = r"D:\D\Ruvision\GFS\Realtime GFS"

logger.info("Folder name from the list")
# Folder name from the list

folder_name = cols[0]

logger.info("Full path to create")
# Full path to create
folder_path = os.path.join(base_path, folder_name)

logger.info("Create the folder")
# Create the folder
os.makedirs(folder_path, exist_ok=True)

print(f"Folder created: {folder_path}")


logger.info("In[6]:")
# In[6]:


forecast_hr = np.arange(15, 85,3)


logger.info("In[7]:")
# In[7]:


forecast_hr


logger.info("In[8]:")
# In[8]:


from datetime import timedelta


logger.info("In[9]:")
# In[9]:


def find_key(dictionary, element):
    for key, value in dictionary.items():
        if element in value:
            return key
    return 'None'


logger.info("In[10]:")
# In[10]:


latbounds = [22.5 - 0.25, 23.5]
lonbounds = [72 , 73 + 0.25]


logger.info("In[11]:")
# In[11]:


time_from_ref = np.arange(15,85,3)


logger.info("In[12]:")
# In[12]:


variable_name = cols[0]


logger.info("In[13]:")
# In[13]:


from datetime import datetime

logger.info("Get the current UTC date and time")
# Get the current UTC date and time
current_utc_datetime = datetime.utcnow().date()
current_utc_datetime


logger.info("In[14]:")
# In[14]:


import os
import requests
from datetime import datetime, timedelta

base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

logger.info("Create a directory to store the downloaded files")
# Create a directory to store the downloaded files
output_directory = f"D:\D\Ruvision\GFS\Realtime GFS\{variable_name}"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

logger.info("Define the date for the forecast (in UTC)")
# Define the date for the forecast (in UTC)
current_date = datetime.utcnow()
logger.info("current_date=current_date - timedelta(days=1)")
#current_date=current_date - timedelta(days=1)

logger.info("List of initialization times (00, 06, 12, and 18 UTC)")
# List of initialization times (00, 06, 12, and 18 UTC)
init_times = [6]

for time in init_times:
logger.info("Loop through forecast hours from 3 to 72 in 3-hour intervals")
    # Loop through forecast hours from 3 to 72 in 3-hour intervals
    for forecast_hour in range(15, 85, 3):
logger.info("Construct the URL for the current forecast hour and initialization time")
        # Construct the URL for the current forecast hour and initialization time
        url = f"{base_url}?dir=%2Fgfs.{current_date.strftime('%Y%m%d')}%2F{time:02d}%2Fatmos&file=gfs.t{time:02d}z.pgrb2.0p25.f{forecast_hour:03d}&var_PRATE=on&lev_surface=on&subregion=&toplat=47&leftlon=55&rightlon=105&bottomlat=0"

logger.info("Extract the filename from the URL")
        # Extract the filename from the URL
        filename = f"gfs.{current_date.strftime('%Y%m%d')}.t{time:02d}z.pgrb2.0p25.f{forecast_hour:03d}"
        filepath = os.path.join(output_directory, filename)

logger.info("Send a GET request to the URL")
        # Send a GET request to the URL
        response = requests.get(url)

logger.info("Check if the request was successful (HTTP status code 200)")
        # Check if the request was successful (HTTP status code 200)
        if response.status_code == 200:
logger.info("Save the downloaded file")
            # Save the downloaded file
            with open(filepath, "wb") as file:
                file.write(response.content)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download: {url}, HTTP status code: {response.status_code}")


logger.info("In[15]:")
# In[15]:


columns_prec = []

for i in cols:
    for time_steps in forecast_hr:
        for j in np.arange(23.5,22.25,-0.25):
            for k in np.arange(72.0,73.25,0.25):
                columns_prec.append(f'{i}_{j}_{k}_{time_steps:03d}')


logger.info("In[16]:")
# In[16]:


import pandas as pd

logger.info("Get today's date at 06:00 UTC")
# Get today's date at 06:00 UTC
ts_06utc = pd.Timestamp.utcnow().normalize() + pd.Timedelta(hours=6)
logger.info("ts_06utc = (pd.Timestamp.utcnow() - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=6)")
#ts_06utc = (pd.Timestamp.utcnow() - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=6)

logger.info("Create DataFrame with just one timestamp (06:00 UTC)")
# Create DataFrame with just one timestamp (06:00 UTC)
data_prec = pd.DataFrame(index=[ts_06utc], columns=columns_prec)

logger.info("Remove timezone info (if any)")
# Remove timezone info (if any)
data_prec.index = data_prec.index.tz_localize(None)


logger.info("In[17]:")
# In[17]:


data_prec


logger.info("In[18]:")
# In[18]:


root_directory = f"D:\D\Ruvision\GFS\Realtime GFS\{variable_name}"
logger.info("Regular expression pattern to match the filenames in the format 'gfs.t00z.pgrb2.0p25.f021'")
# Regular expression pattern to match the filenames in the format "gfs.t00z.pgrb2.0p25.f021"
filename_pattern = r'gfs\.\d{8}\.t\d{2}z\.pgrb2\.0p25\.f\d{3}$'

logger.info("Dictionary to store the filenames grouped by directory")
# Dictionary to store the filenames grouped by directory
directory_names = {}

logger.info("Function to check if the filename matches the expected pattern")
# Function to check if the filename matches the expected pattern
def is_grib2_file(filename):
    return re.match(filename_pattern, filename)

logger.info("Walk through the root directory and its subdirectories")
# Walk through the root directory and its subdirectories
for dirpath, dirnames, filenames in os.walk(root_directory):
    x = str(dirpath).split(os.sep)[-1]
    directory_names[x] = []

logger.info("Filter the filenames based on the expected pattern and store them in the dictionary")
    # Filter the filenames based on the expected pattern and store them in the dictionary
    for filename in filenames:
        if is_grib2_file(filename):
            directory_names[x].append(filename)

logger.info("Print the filenames of the GRIB2 files in each directory")
# Print the filenames of the GRIB2 files in each directory
for directory, filenames in directory_names.items():
    print(f"Directory: {directory}")
    for filename in filenames:
        print(filename)
    print()  # Add an empty line to separate directories


logger.info("In[19]:")
# In[19]:


logger.info("Open the GRIB file")
# Open the GRIB file
filename = f"D:\D\Ruvision\GFS\Realtime GFS\{variable_name}\gfs.20250722.t06z.pgrb2.0p25.f015"
grbs = pygrib.open(filename)

logger.info("Print information about each GRIB message (parameter)")
# Print information about each GRIB message (parameter)
for grb in grbs:
    print(f"Parameter Name: {grb.name}")
    print(f"Level: {grb.level}")
    print(f"Units: {grb.units}")
    print(f"Values: {grb.values}")
    print(f"Grid Shape: {grb.values.shape}")
    print("----------")

logger.info("Close the GRIB file")
# Close the GRIB file
grbs.close()


logger.info("In[20]:")
# In[20]:


counter=0
for time_step in data_prec.index:

    year = time_step.year
    month = time_step.month
    day = time_step.day

    ref_time = time_step.hour

    date_temp = pd.date_range(start = time_step + timedelta(hours = 15), end = time_step + timedelta(hours = 84) , freq = '3h')
    col_temp = np.arange(0,25)

    data_temp = pd.DataFrame(index = date_temp, columns=col_temp)

    for time_lag in time_from_ref:

        filename = f'gfs.{year}{month:02d}{day:02d}.t06z.pgrb2.0p25.f{time_lag:03d}'

        grib =f'{root_directory}/{filename}'
        grbs=pygrib.open(grib)
        variable = 'Precipitation rate'

        for grb in grbs.select(name=variable):
            temp=grb.values
logger.info("reshaped_data= data.reshape(1, 9, 9)")
            #reshaped_data= data.reshape(1, 9, 9)
logger.info("data=reshaped_data")
            #data=reshaped_data
            lats, lons= grb.latlons()
            lats_reshaped = lats[:,0]  # Reshape latitudes to (189,)
            reversed_arr = lats_reshaped[::-1]
            lons_reshaped = lons[0,:]  # Reshape longitudes to (,201)
            lats=reversed_arr
            lons=lons_reshaped
            parameter_name = grb.name
            level_type = grb.typeOfLevel
            parameter_units = grb.parameterUnits
            level = grb.level
            forecast_time = grb.forecastTime
            valid_date = grb.validDate

logger.info("latitude lower and upper index")
        # latitude lower and upper index
        latli = np.argmin( np.abs( reversed_arr - latbounds[1] ) )
        latui = np.argmin( np.abs( reversed_arr - latbounds[0] ) ) 

logger.info("longitude lower and upper index")
        # longitude lower and upper index
        lonli = np.argmin( np.abs( lons_reshaped- lonbounds[0] ) )
        lonui = np.argmin( np.abs( lons_reshaped - lonbounds[1] ) )  

        time = pd.to_datetime(f'{year}-{month}-{day}') + timedelta(hours = int(int(ref_time) + int(time_lag)))
        data= temp[latli:latui, lonli:lonui][::-1]
        time_prev = time - timedelta(hours = 3)

        if ((int(time_lag)%6) == 0):
            if len(np.ravel(data)) == 25:
                data_temp.loc[time][0:25] =  (np.ravel(data)*21600) - np.ravel(data_temp.loc[time_prev][0:25])

        elif (((int(time_lag) % 6) != 0) & ((int(time_lag) % 3) == 0)):
            if len(np.ravel(data)) == 25:
                data_temp.loc[time][0:25] = (np.ravel(data)*10800)
        else:
            print(filename)

    data_prec.loc[time_step][0:600] = np.ravel(data_temp)

    counter += 1
    if (counter)%100 == 0:
        print(f'Loop {counter} Done!')


logger.info("In[21]:")
# In[21]:


data_prec.isnull().sum().max()


logger.info("In[22]:")
# In[22]:


data_prec


logger.info("In[23]:")
# In[23]:


data_final_interpolated1=data_prec


logger.info("In[24]:")
# In[24]:


data_final_interpolated1 = data_final_interpolated1.shift(freq=pd.Timedelta(hours=5, minutes=30))


logger.info("In[25]:")
# In[25]:


data_final_interpolated1 = data_final_interpolated1.rename_axis('DateTime')


logger.info("In[26]:")
# In[26]:


extracted_rows = data_final_interpolated1[data_final_interpolated1.index.time == pd.Timestamp("11:30").time()]


logger.info("In[27]:")
# In[27]:


data_prec_lead_day_1 = extracted_rows.loc[:, ~extracted_rows.columns.str.contains('|'.join(['039', '042', '045', '048', '051', '054', '057', '060', '063', '066', '069', '072', '075', '078', '081', '084']))]
data_prec_lead_day_2 = extracted_rows.loc[:, ~extracted_rows.columns.str.contains('|'.join(['015', '018', '021', '024', '027', '030', '033', '036', '063', '066', '069', '072', '075', '078', '081', '084']))]
data_prec_lead_day_3 = extracted_rows.loc[:, ~extracted_rows.columns.str.contains('|'.join(['015', '018', '021', '024', '027', '030', '033', '036', '039', '042', '045', '048', '051', '054', '057', '060']))]


logger.info("In[28]:")
# In[28]:


columns_prec = []
for i in cols:
    for j in np.arange(23.5,22.25,-0.25):
        for k in np.arange(72.0,73.25,0.25):
            columns_prec.append(f'{i}_{j}_{k}')


logger.info("In[29]:")
# In[29]:


start_date = extracted_rows.index[0]
end_date = extracted_rows.index[-1]


logger.info("In[30]:")
# In[30]:


logger.info("Update the start and end dates")
# Update the start and end dates
updated_start_date_1 = pd.to_datetime(start_date) + pd.Timedelta(hours=15)
updated_end_date_1 = pd.to_datetime(end_date) + pd.Timedelta(hours=36)
logger.info("Update the start and end dates")
# Update the start and end dates
updated_start_date_2 = pd.to_datetime(start_date) + pd.Timedelta(hours=39)
updated_end_date_2 = pd.to_datetime(end_date) + pd.Timedelta(hours=60)
logger.info("Update the start and end dates")
# Update the start and end dates
updated_start_date_3 = pd.to_datetime(start_date) + pd.Timedelta(hours=63)
updated_end_date_3 = pd.to_datetime(end_date) + pd.Timedelta(hours=84)

data_prec_1 = pd.DataFrame(index = pd.date_range(start=updated_start_date_1, end=updated_end_date_1, freq = '3h'), columns = columns_prec)
data_prec_2 = pd.DataFrame(index = pd.date_range(start=updated_start_date_2, end=updated_end_date_2, freq = '3h'), columns = columns_prec)
data_prec_3 = pd.DataFrame(index = pd.date_range(start=updated_start_date_3, end=updated_end_date_3, freq = '3h'), columns = columns_prec)


logger.info("In[31]:")
# In[31]:


selected_rows_1 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('02:30:00').time()]
selected_rows_2 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('05:30:00').time()]
selected_rows_3 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('08:30:00').time()]
selected_rows_4 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('11:30:00').time()]
selected_rows_5 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('14:30:00').time()]
selected_rows_6 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('17:30:00').time()]
selected_rows_7 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('20:30:00').time()]
selected_rows_8 = data_prec_1.loc[data_prec_1.index.time == pd.Timestamp('23:30:00').time()]

selected_rows_9 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('02:30:00').time()]
selected_rows_10 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('05:30:00').time()]
selected_rows_11 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('08:30:00').time()]
selected_rows_12 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('11:30:00').time()]
selected_rows_13 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('14:30:00').time()]
selected_rows_14 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('17:30:00').time()]
selected_rows_15 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('20:30:00').time()]
selected_rows_16 = data_prec_2.loc[data_prec_2.index.time == pd.Timestamp('23:30:00').time()]

selected_rows_17 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('02:30:00').time()]
selected_rows_18 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('05:30:00').time()]
selected_rows_19 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('08:30:00').time()]
selected_rows_20 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('11:30:00').time()]
selected_rows_21 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('14:30:00').time()]
selected_rows_22= data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('17:30:00').time()]
selected_rows_23 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('20:30:00').time()]
selected_rows_24 = data_prec_3.loc[data_prec_3.index.time == pd.Timestamp('23:30:00').time()]


logger.info("In[32]:")
# In[32]:


x1=data_prec_lead_day_1.iloc[:, :25]
x2=data_prec_lead_day_1.iloc[:, 25:50]
x3=data_prec_lead_day_1.iloc[:, 50:75]
x4=data_prec_lead_day_1.iloc[:, 75:100]
x5=data_prec_lead_day_1.iloc[:, 100:125]
x6=data_prec_lead_day_1.iloc[:, 125:150]
x7=data_prec_lead_day_1.iloc[:, 150:175]
x8=data_prec_lead_day_1.iloc[:, 175:]

x9=data_prec_lead_day_2.iloc[:, :25]
x10=data_prec_lead_day_2.iloc[:, 25:50]
x11=data_prec_lead_day_2.iloc[:, 50:75]
x12=data_prec_lead_day_2.iloc[:, 75:100]
x13=data_prec_lead_day_2.iloc[:, 100:125]
x14=data_prec_lead_day_2.iloc[:, 125:150]
x15=data_prec_lead_day_2.iloc[:, 150:175]
x16=data_prec_lead_day_2.iloc[:, 175:]

x17=data_prec_lead_day_3.iloc[:, :25]
x18=data_prec_lead_day_3.iloc[:, 25:50]
x19=data_prec_lead_day_3.iloc[:, 50:75]
x20=data_prec_lead_day_3.iloc[:, 75:100]
x21=data_prec_lead_day_3.iloc[:, 100:125]
x22=data_prec_lead_day_3.iloc[:, 125:150]
x23=data_prec_lead_day_3.iloc[:, 150:175]
x24=data_prec_lead_day_3.iloc[:, 175:]


logger.info("In[33]:")
# In[33]:


selected_rows_1.loc[:, :] = x1.values
selected_rows_2.loc[:, :] = x2.values
selected_rows_3.loc[:, :] = x3.values
selected_rows_4.loc[:, :] = x4.values
selected_rows_5.loc[:, :] = x5.values
selected_rows_6.loc[:, :] = x6.values
selected_rows_7.loc[:, :] = x7.values
selected_rows_8.loc[:, :] = x8.values
selected_rows_9.loc[:, :] = x9.values
selected_rows_10.loc[:, :] = x10.values
selected_rows_11.loc[:, :] = x11.values
selected_rows_12.loc[:, :] = x12.values
selected_rows_13.loc[:, :] = x13.values
selected_rows_14.loc[:, :] = x14.values
selected_rows_15.loc[:, :] = x15.values
selected_rows_16.loc[:, :] = x16.values
selected_rows_17.loc[:, :] = x17.values
selected_rows_18.loc[:, :] = x18.values
selected_rows_19.loc[:, :] = x19.values
selected_rows_20.loc[:, :] = x20.values
selected_rows_21.loc[:, :] = x21.values
selected_rows_22.loc[:, :] = x22.values
selected_rows_23.loc[:, :] = x23.values
selected_rows_24.loc[:, :] = x24.values


logger.info("In[34]:")
# In[34]:


merged_df_1 = pd.concat([selected_rows_1, selected_rows_2, selected_rows_3, selected_rows_4, 
                       selected_rows_5, selected_rows_6, selected_rows_7, selected_rows_8], axis=0)

merged_df_2 = pd.concat([selected_rows_9, selected_rows_10, selected_rows_11, selected_rows_12, 
                       selected_rows_13, selected_rows_14, selected_rows_15, selected_rows_16], axis=0)

merged_df_3 = pd.concat([selected_rows_17, selected_rows_18, selected_rows_19, selected_rows_20, 
                       selected_rows_21, selected_rows_22, selected_rows_23, selected_rows_24], axis=0)

merged_df_1 = merged_df_1.rename_axis('DateTime')
merged_df_1.reset_index('DateTime', inplace=True)
sorted_df_1 = merged_df_1.sort_values(by='DateTime', ascending=True)
sorted_df_1.set_index('DateTime', inplace=True)
data_X_Lead_Day_1=sorted_df_1


merged_df_2 = merged_df_2.rename_axis('DateTime')
merged_df_2.reset_index('DateTime', inplace=True)
sorted_df_2 = merged_df_2.sort_values(by='DateTime', ascending=True)
sorted_df_2.set_index('DateTime', inplace=True)
data_X_Lead_Day_2=sorted_df_2

merged_df_3 = merged_df_3.rename_axis('DateTime')
merged_df_3.reset_index('DateTime', inplace=True)
sorted_df_3 = merged_df_3.sort_values(by='DateTime', ascending=True)
sorted_df_3.set_index('DateTime', inplace=True)
data_X_Lead_Day_3=sorted_df_3

group_idx_1 = (data_X_Lead_Day_1.index.to_series().reset_index(drop=True).index // 8)
group_idx_2 = (data_X_Lead_Day_2.index.to_series().reset_index(drop=True).index // 8)
group_idx_3 = (data_X_Lead_Day_3.index.to_series().reset_index(drop=True).index // 8)

summed_data_1 = data_X_Lead_Day_1.groupby(group_idx_1).sum()
summed_data_2 = data_X_Lead_Day_2.groupby(group_idx_2).sum()
summed_data_3 = data_X_Lead_Day_3.groupby(group_idx_3).sum()

logger.info("Get today's date normalized to midnight")
# Get today's date normalized to midnight
today = pd.Timestamp.today().normalize()
logger.info("today = pd.Timestamp.today().normalize()- pd.Timedelta(days=1)")
#today = pd.Timestamp.today().normalize()- pd.Timedelta(days=1)

logger.info("Generate dates for tomorrow, day after, and two days after — all at 23:30")
# Generate dates for tomorrow, day after, and two days after — all at 23:30
date1 = today + pd.Timedelta(days=1, hours=23, minutes=30)
date2 = today + pd.Timedelta(days=2, hours=23, minutes=30)
date3 = today + pd.Timedelta(days=3, hours=23, minutes=30)

logger.info("Create DataFrames")
# Create DataFrames
data_prec_1 = pd.DataFrame(index=pd.date_range(start=date1, end=date1, freq='24h'), columns=summed_data_1.columns)
data_prec_2 = pd.DataFrame(index=pd.date_range(start=date2, end=date2, freq='24h'), columns=summed_data_2.columns)
data_prec_3 = pd.DataFrame(index=pd.date_range(start=date3, end=date3, freq='24h'), columns=summed_data_3.columns)

summed_data_1['DateTime']= data_prec_1.index
summed_data_2['DateTime']= data_prec_2.index
summed_data_3['DateTime']= data_prec_3.index

summed_data_1.set_index('DateTime', inplace=True)
summed_data_2.set_index('DateTime', inplace=True)
summed_data_3.set_index('DateTime', inplace=True)


logger.info("In[35]:")
# In[35]:


summed_data_2


logger.info("In[36]:")
# In[36]:


X1 = pd.read_excel(f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 1_daily basis.xlsx")
X2 = pd.read_excel(f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 2_daily basis.xlsx")
X3 = pd.read_excel(f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 3_daily basis.xlsx")


logger.info("In[37]:")
# In[37]:


X1.set_index('DateTime', inplace=True)
X2.set_index('DateTime', inplace=True)
X3.set_index('DateTime', inplace=True)


logger.info("In[38]:")
# In[38]:


Data_X1= pd.concat([X1, summed_data_1], axis=0)
Data_X2= pd.concat([X2, summed_data_2], axis=0)
Data_X3= pd.concat([X3, summed_data_3], axis=0)


logger.info("In[39]:")
# In[39]:


Data_X2


logger.info("In[40]:")
# In[40]:


filename_1 = f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 1_daily basis.xlsx"
filename_2 = f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 2_daily basis.xlsx"
filename_3 = f"D:/D/Ruvision/GFS/Realtime GFS/{variable_name}_Ahmedabad_Lead Day 3_daily basis.xlsx"

logger.info("Save the DataFrame")
# Save the DataFrame
Data_X1.to_excel(filename_1)
Data_X2.to_excel(filename_2)
Data_X3.to_excel(filename_3)


logger.info("In[ ]:")
# In[ ]:





logger.info("In[ ]:")
# In[ ]:





logger.info("In[ ]:")
# In[ ]:
# === End: Pygrib_GFS Rain Data Automated Download and Preprocessing_RUVISION_FINAL_22-07-2025.ipynb ===


# === Begin: Realtime Results Generation_Final.ipynb ===

logger.info("!/usr/bin/env python")
#!/usr/bin/env python
logger.info("coding: utf-8")
# coding: utf-8

logger.info("In[95]:")
# In[95]:


import pandas as pd
import numpy as np


logger.info("In[96]:")
# In[96]:


APCP= pd.read_excel(r'D:\D\Ruvision\GFS\Realtime GFS\PREC_Ahmedabad_Lead Day 3_daily basis.xlsx')


logger.info("In[97]:")
# In[97]:


APCP.set_index('DateTime', inplace=True)


logger.info("In[98]:")
# In[98]:


u= pd.read_excel(r'D:\D\Ruvision\GFS\Realtime GFS\U1000_Ahmedabad_Lead Day 3_daily basis.xlsx')


logger.info("In[99]:")
# In[99]:


u.set_index('DateTime', inplace=True)


logger.info("In[100]:")
# In[100]:


v= pd.read_excel(r'D:\D\Ruvision\GFS\Realtime GFS\V1000_Ahmedabad_Lead Day 3_daily basis.xlsx')


logger.info("In[101]:")
# In[101]:


v.set_index('DateTime', inplace=True)


logger.info("In[102]:")
# In[102]:


Data_X= pd.concat([u, v, APCP], axis=1)


logger.info("In[103]:")
# In[103]:


Data_X


logger.info("In[104]:")
# In[104]:


Data_X.isnull().sum().sum()


logger.info("In[105]:")
# In[105]:


Data_X_future= Data_X.iloc[3639:, :]


logger.info("In[106]:")
# In[106]:


Data_X_future


logger.info("In[107]:")
# In[107]:


from sklearn.preprocessing import MinMaxScaler

logger.info("Initialize the MinMaxScaler")
# Initialize the MinMaxScaler
scaler = MinMaxScaler()

logger.info("Fit and transform the data")
# Fit and transform the data
data_X_scaled = scaler.fit_transform(Data_X)

logger.info("Convert the scaled array back to a DataFrame")
# Convert the scaled array back to a DataFrame
data_X_normalized = pd.DataFrame(data_X_scaled, columns=Data_X.columns)

data_X_normalized


logger.info("In[108]:")
# In[108]:


from sklearn.decomposition import PCA


logger.info("In[109]:")
# In[109]:


pca = PCA(.99)


logger.info("In[110]:")
# In[110]:


pca.fit(data_X_normalized)


logger.info("In[111]:")
# In[111]:


pca.n_components_


logger.info("In[112]:")
# In[112]:


PCA_data_X= pca.transform(data_X_normalized)


logger.info("In[113]:")
# In[113]:


transformed_data_X = pd.DataFrame(index = Data_X.index, data = PCA_data_X)


logger.info("In[114]:")
# In[114]:


transformed_data_X


logger.info("In[115]:")
# In[115]:


transformed_data_X.to_pickle(r'D:\D\Ruvision\GFS\Realtime GFS\Pickle\transformed_dataX_LD_3_U1000_V1000_PREC.pkl')


logger.info("In[116]:")
# In[116]:


import os
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import re

# === FUNCTIONS ===

def load_data(pkl_path, excel_df):
    X_full = pd.read_pickle(pkl_path)
    Y = excel_df.set_index('DateTime')
    Y.index = pd.to_datetime(Y.index)
    X = X_full.iloc[:len(Y)]
    X.index = Y.index
    return X, Y, X_full

def split_data(X, Y, split_index=3269):
    X_train = X.iloc[:split_index]
    Y_train = Y.iloc[:split_index]
    X_test = X.iloc[split_index+3:]
    Y_test = Y.iloc[split_index+3:]
    return X_train, X_test, Y_train, Y_test

def binary_glm(X_train, Y_train):
    yb_train = (Y_train > 0).astype(int)
    glm = sm.GLM(yb_train, sm.add_constant(X_train), family=sm.families.Binomial()).fit()
    yfit_train = glm.predict(sm.add_constant(X_train))
    return yfit_train

def quantile_regression(X, Y, q):
    mod = sm.QuantReg(Y, sm.add_constant(X)).fit(q=q)
    return mod.params

def reconstruct_rain(X, coefs):
    return X @ coefs[1:] + coefs[0]

def cqm_pipeline(X_train, Y_train, X_test, thresholds):
    yfit_train = binary_glm(X_train, Y_train)
    coefs2 = {}
    rain_final = {}

    for label, t in thresholds.items():
        q_val = float(label.replace('p', '')) / 100
        mask1 = yfit_train > t
        X1, Y1 = X_train[mask1], Y_train[mask1]
        if X1.empty or Y1.empty:
            continue
        coef1 = quantile_regression(X1, Y1, q_val)
        rain1 = reconstruct_rain(X1, coef1)
        mask2 = rain1 > 0
        X2, Y2 = X1[mask2], Y1[mask2]
        if X2.empty or Y2.empty:
            continue
        coef2 = quantile_regression(X2, Y2, q_val)
        coefs2[label] = coef2
        rain_test = reconstruct_rain(X_test, coef2)
        rain_final[label] = np.maximum(rain_test, 0)

    return coefs2, rain_final

# === CONFIG ===

pca_folder = r'D:\D\Ruvision\GFS\Realtime GFS\Pickle'
obs_excel = pd.read_excel(
    'C:/Users/Angshudeep Majumdar/Documents/Angshudeep Lappy/Ruvision/IMD_2015-2024_Daily Data_0.25 resolution Rain_GFS_LD_3.xlsx'
)
output_base = r'D:\D\Ruvision\GFS\Realtime GFS\Pickle'

thresholds = {'80p': 0.2, '85p': 0.15, '90p': 0.1, '95p': 0.05, '99p': 0.01}

# === MAIN LOOP ===

pattern = r'transformed_dataX_LD_3_(.+)\.pkl'

for pkl in glob.glob(os.path.join(pca_folder, '*.pkl')):
    match = re.search(pattern, os.path.basename(pkl))
    if not match:
        print(f"Skipping file (pattern not matched): {pkl}")
        continue

    var_comb = match.group(1)
    out_dir = os.path.join(output_base, var_comb)
    os.makedirs(out_dir, exist_ok=True)

logger.info("Load data")
    # Load data
    X, Y_df, X_full = load_data(pkl, obs_excel)

    if 'Prec_23.0_72.5' not in Y_df.columns:
        print(f"Column 'Prec_23.0_72.5' not found for {var_comb}. Skipping.")
        continue

    Y = Y_df['Prec_23.0_72.5']

logger.info("Split for training")
    # Split for training
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

logger.info("Train CQM")
    # Train CQM
    coefs2, _ = cqm_pipeline(X_train, Y_train, X_test, thresholds)

    # === FUTURE DATA ===
    X_future = X_full.iloc[len(Y):]
    if X_future.empty:
        print(f"No future data for {var_comb}. Skipping.")
        continue

    last_y_date = Y.index[-1]
    future_dates = pd.date_range(start=last_y_date + pd.Timedelta(days=1), periods=len(X_future), freq='D')
    X_future.index = future_dates

logger.info("Predict for future using trained coefs")
    # Predict for future using trained coefs
    rain_future = {}
    for label, coefs in coefs2.items():
        rain_pred = reconstruct_rain(X_future, coefs)
        rain_future[label] = np.maximum(rain_pred, 0)

logger.info("Extract JJAS 2025")
    # Extract JJAS 2025
    jjas_start_2025 = pd.Timestamp('2025-06-03')
    jjas_end_2025 = pd.Timestamp('2025-09-30')
    mask_jjas_2025 = (X_future.index >= jjas_start_2025) & (X_future.index <= jjas_end_2025)

    rain_future_jjas_2025 = {
        k: pd.Series(v, index=X_future.index)[mask_jjas_2025]
        for k, v in rain_future.items()
    }

    # === CREATE DATAFRAME OF FUTURE PREDICTIONS FOR JJAS 2025 ===
    df_jjas_2025 = pd.DataFrame({
        q_label: series
        for q_label, series in rain_future_jjas_2025.items()
    })

logger.info("Save to Excel (remove 00:00:00 timestamp)")
    # Save to Excel (remove 00:00:00 timestamp)
    df_jjas_2025.index = df_jjas_2025.index.date
    csv_path = os.path.join(out_dir, f'CQM_QuantileForecast_{var_comb}_2025JJAS_LD_3.xlsx')
    df_jjas_2025.to_excel(csv_path, index_label="Date")
    print(f"Forecast DataFrame (with GFS) saved to: {csv_path}")

    # === BAR PLOT: LAST 3 DAYS OF JJAS 2025 for 80th quantile only ===
    q_label = '80p'
    if q_label not in rain_future_jjas_2025:
        print(f"Missing {q_label} for plotting {var_comb}, skipping plot.")
        continue

    pred_series = rain_future_jjas_2025[q_label]
    if pred_series.empty:
        print(f"{q_label} series is empty for {var_comb}, skipping plot.")
        continue

    last_day = pred_series.index[-1:]
    pred_series_3rd_lead = pred_series.loc[last_day]
pred_series_3rd_lead


logger.info("In[117]:")
# In[117]:


logger.info("Lead Day 2")
#Lead Day 2


logger.info("In[118]:")
# In[118]:


APCP= pd.read_excel(r'D:\D\Ruvision\GFS\Realtime GFS\PREC_Ahmedabad_Lead Day 2_daily basis.xlsx')


logger.info("In[119]:")
# In[119]:


APCP.set_index('DateTime', inplace=True)


logger.info("In[120]:")
# In[120]:


u= pd.read_excel(r'D:\D\Ruvision\GFS\Realtime GFS\U1000_Ahmedabad_Lead Day 2_daily basis.xlsx')


logger.info("In[121]:")
# In[121]:


u.set_index('DateTime', inplace=True)


logger.info("In[122]:")
# In[122]:


v= pd.read_excel(r'D:\D\Ruvision\GFS\Realtime GFS\V1000_Ahmedabad_Lead Day 2_daily basis.xlsx')


logger.info("In[123]:")
# In[123]:


v.set_index('DateTime', inplace=True)


logger.info("In[124]:")
# In[124]:


Data_X= pd.concat([u, v, APCP], axis=1)


logger.info("In[125]:")
# In[125]:


Data_X


logger.info("In[126]:")
# In[126]:


Data_X.isnull().sum().sum()


logger.info("In[127]:")
# In[127]:


Data_X_future= Data_X.iloc[3639:, :]


logger.info("In[128]:")
# In[128]:


logger.info("Initialize the MinMaxScaler")
# Initialize the MinMaxScaler
scaler = MinMaxScaler()

logger.info("Fit and transform the data")
# Fit and transform the data
data_X_scaled = scaler.fit_transform(Data_X)

logger.info("Convert the scaled array back to a DataFrame")
# Convert the scaled array back to a DataFrame
data_X_normalized = pd.DataFrame(data_X_scaled, columns=Data_X.columns)

data_X_normalized


logger.info("In[129]:")
# In[129]:


pca = PCA(.99)


logger.info("In[130]:")
# In[130]:


pca.fit(data_X_normalized)


logger.info("In[131]:")
# In[131]:


pca.n_components_


logger.info("In[132]:")
# In[132]:


PCA_data_X= pca.transform(data_X_normalized)


logger.info("In[133]:")
# In[133]:


transformed_data_X = pd.DataFrame(index = Data_X.index, data = PCA_data_X)


logger.info("In[134]:")
# In[134]:


transformed_data_X


logger.info("In[135]:")
# In[135]:


transformed_data_X.to_pickle(r'D:\D\Ruvision\GFS\Realtime GFS\Pickle\transformed_dataX_LD_2_U1000_V1000_PREC.pkl')


logger.info("In[136]:")
# In[136]:


import os
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import re

# === FUNCTIONS ===

def load_data(pkl_path, excel_df):
    X_full = pd.read_pickle(pkl_path)
    Y = excel_df.set_index('DateTime')
    Y.index = pd.to_datetime(Y.index)
    X = X_full.iloc[:len(Y)]
    X.index = Y.index
    return X, Y, X_full

def split_data(X, Y, split_index=3269):
    X_train = X.iloc[:split_index]
    Y_train = Y.iloc[:split_index]
    X_test = X.iloc[split_index+3:]
    Y_test = Y.iloc[split_index+3:]
    return X_train, X_test, Y_train, Y_test

def binary_glm(X_train, Y_train):
    yb_train = (Y_train > 0).astype(int)
    glm = sm.GLM(yb_train, sm.add_constant(X_train), family=sm.families.Binomial()).fit()
    yfit_train = glm.predict(sm.add_constant(X_train))
    return yfit_train

def quantile_regression(X, Y, q):
    mod = sm.QuantReg(Y, sm.add_constant(X)).fit(q=q)
    return mod.params

def reconstruct_rain(X, coefs):
    return X @ coefs[1:] + coefs[0]

def cqm_pipeline(X_train, Y_train, X_test, thresholds):
    yfit_train = binary_glm(X_train, Y_train)
    coefs2 = {}
    rain_final = {}

    for label, t in thresholds.items():
        q_val = float(label.replace('p', '')) / 100
        mask1 = yfit_train > t
        X1, Y1 = X_train[mask1], Y_train[mask1]
        if X1.empty or Y1.empty:
            continue
        coef1 = quantile_regression(X1, Y1, q_val)
        rain1 = reconstruct_rain(X1, coef1)
        mask2 = rain1 > 0
        X2, Y2 = X1[mask2], Y1[mask2]
        if X2.empty or Y2.empty:
            continue
        coef2 = quantile_regression(X2, Y2, q_val)
        coefs2[label] = coef2
        rain_test = reconstruct_rain(X_test, coef2)
        rain_final[label] = np.maximum(rain_test, 0)

    return coefs2, rain_final

# === CONFIG ===

pca_folder = r'D:\D\Ruvision\GFS\Realtime GFS\Pickle'
obs_excel = pd.read_excel(
    'C:/Users/Angshudeep Majumdar/Documents/Angshudeep Lappy/Ruvision/IMD_2015-2024_Daily Data_0.25 resolution Rain_GFS_LD_2.xlsx'
)
output_base = r'D:\D\Ruvision\GFS\Realtime GFS\Pickle'

thresholds = {'80p': 0.2, '85p': 0.15, '90p': 0.1, '95p': 0.05, '99p': 0.01}

# === MAIN LOOP ===

pattern = r'transformed_dataX_LD_2_(.+)\.pkl'

for pkl in glob.glob(os.path.join(pca_folder, '*.pkl')):
    match = re.search(pattern, os.path.basename(pkl))
    if not match:
        print(f"Skipping file (pattern not matched): {pkl}")
        continue

    var_comb = match.group(1)
    out_dir = os.path.join(output_base, var_comb)
    os.makedirs(out_dir, exist_ok=True)

logger.info("Load data")
    # Load data
    X, Y_df, X_full = load_data(pkl, obs_excel)

    if 'Prec_23.0_72.5' not in Y_df.columns:
        print(f"Column 'Prec_23.0_72.5' not found for {var_comb}. Skipping.")
        continue

    Y = Y_df['Prec_23.0_72.5']

logger.info("Split for training")
    # Split for training
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

logger.info("Train CQM")
    # Train CQM
    coefs2, _ = cqm_pipeline(X_train, Y_train, X_test, thresholds)

    # === FUTURE DATA ===
    X_future = X_full.iloc[len(Y):]
    if X_future.empty:
        print(f"No future data for {var_comb}. Skipping.")
        continue

    last_y_date = Y.index[-1]
    future_dates = pd.date_range(start=last_y_date + pd.Timedelta(days=1), periods=len(X_future), freq='D')
    X_future.index = future_dates

logger.info("Predict for future using trained coefs")
    # Predict for future using trained coefs
    rain_future = {}
    for label, coefs in coefs2.items():
        rain_pred = reconstruct_rain(X_future, coefs)
        rain_future[label] = np.maximum(rain_pred, 0)

logger.info("Extract JJAS 2025")
    # Extract JJAS 2025
    jjas_start_2025 = pd.Timestamp('2025-06-02')
    jjas_end_2025 = pd.Timestamp('2025-09-30')
    mask_jjas_2025 = (X_future.index >= jjas_start_2025) & (X_future.index <= jjas_end_2025)

    rain_future_jjas_2025 = {
        k: pd.Series(v, index=X_future.index)[mask_jjas_2025]
        for k, v in rain_future.items()
    }

    # === CREATE DATAFRAME OF FUTURE PREDICTIONS FOR JJAS 2025 ===
    df_jjas_2025 = pd.DataFrame({
        q_label: series
        for q_label, series in rain_future_jjas_2025.items()
    })

logger.info("Save to Excel (remove 00:00:00 timestamp)")
    # Save to Excel (remove 00:00:00 timestamp)
    df_jjas_2025.index = df_jjas_2025.index.date
    csv_path = os.path.join(out_dir, f'CQM_QuantileForecast_{var_comb}_2025JJAS_LD_2.xlsx')
    df_jjas_2025.to_excel(csv_path, index_label="Date")
    print(f"Forecast DataFrame (with GFS) saved to: {csv_path}")

    # === BAR PLOT: LAST 3 DAYS OF JJAS 2025 for 80th quantile only ===
    q_label = '80p'
    if q_label not in rain_future_jjas_2025:
        print(f"Missing {q_label} for plotting {var_comb}, skipping plot.")
        continue

    pred_series = rain_future_jjas_2025[q_label]
    if pred_series.empty:
        print(f"{q_label} series is empty for {var_comb}, skipping plot.")
        continue

    last_day = pred_series.index[-1:]
    pred_series_2nd_day_lead = pred_series.loc[last_day]
pred_series_2nd_day_lead


logger.info("In[137]:")
# In[137]:


logger.info("Lead Day 1")
#Lead Day 1


logger.info("In[138]:")
# In[138]:


APCP= pd.read_excel(r'D:\D\Ruvision\GFS\Realtime GFS\PREC_Ahmedabad_Lead Day 1_daily basis.xlsx')


logger.info("In[139]:")
# In[139]:


APCP.set_index('DateTime', inplace=True)


logger.info("In[140]:")
# In[140]:


u= pd.read_excel(r'D:\D\Ruvision\GFS\Realtime GFS\U1000_Ahmedabad_Lead Day 1_daily basis.xlsx')


logger.info("In[141]:")
# In[141]:


u.set_index('DateTime', inplace=True)


logger.info("In[142]:")
# In[142]:


v= pd.read_excel(r'D:\D\Ruvision\GFS\Realtime GFS\V1000_Ahmedabad_Lead Day 1_daily basis.xlsx')


logger.info("In[143]:")
# In[143]:


v.set_index('DateTime', inplace=True)


logger.info("In[144]:")
# In[144]:


Data_X= pd.concat([u, v, APCP], axis=1)


logger.info("In[145]:")
# In[145]:


Data_X


logger.info("In[146]:")
# In[146]:


Data_X.isnull().sum().sum()


logger.info("In[147]:")
# In[147]:


Data_X_future= Data_X.iloc[3639:, :]


logger.info("In[148]:")
# In[148]:


Data_X_future


logger.info("In[149]:")
# In[149]:


from sklearn.preprocessing import MinMaxScaler

logger.info("Initialize the MinMaxScaler")
# Initialize the MinMaxScaler
scaler = MinMaxScaler()

logger.info("Fit and transform the data")
# Fit and transform the data
data_X_scaled = scaler.fit_transform(Data_X)

logger.info("Convert the scaled array back to a DataFrame")
# Convert the scaled array back to a DataFrame
data_X_normalized = pd.DataFrame(data_X_scaled, columns=Data_X.columns)

data_X_normalized


logger.info("In[150]:")
# In[150]:


pca = PCA(.99)


logger.info("In[151]:")
# In[151]:


pca.fit(data_X_normalized)


logger.info("In[152]:")
# In[152]:


pca.n_components_


logger.info("In[153]:")
# In[153]:


PCA_data_X= pca.transform(data_X_normalized)


logger.info("In[154]:")
# In[154]:


transformed_data_X = pd.DataFrame(index = Data_X.index, data = PCA_data_X)


logger.info("In[155]:")
# In[155]:


transformed_data_X


logger.info("In[156]:")
# In[156]:


transformed_data_X.to_pickle(r'D:\D\Ruvision\GFS\Realtime GFS\Pickle\transformed_dataX_LD_1_U1000_V1000_PREC.pkl')


logger.info("In[157]:")
# In[157]:


import os
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import re

# === FUNCTIONS ===

def load_data(pkl_path, excel_df):
    X_full = pd.read_pickle(pkl_path)
    Y = excel_df.set_index('DateTime')
    Y.index = pd.to_datetime(Y.index)
    X = X_full.iloc[:len(Y)]
    X.index = Y.index
    return X, Y, X_full

def split_data(X, Y, split_index=3269):
    X_train = X.iloc[:split_index]
    Y_train = Y.iloc[:split_index]
    X_test = X.iloc[split_index+3:]
    Y_test = Y.iloc[split_index+3:]
    return X_train, X_test, Y_train, Y_test

def binary_glm(X_train, Y_train):
    yb_train = (Y_train > 0).astype(int)
    glm = sm.GLM(yb_train, sm.add_constant(X_train), family=sm.families.Binomial()).fit()
    yfit_train = glm.predict(sm.add_constant(X_train))
    return yfit_train

def quantile_regression(X, Y, q):
    mod = sm.QuantReg(Y, sm.add_constant(X)).fit(q=q)
    return mod.params

def reconstruct_rain(X, coefs):
    return X @ coefs[1:] + coefs[0]

def cqm_pipeline(X_train, Y_train, X_test, thresholds):
    yfit_train = binary_glm(X_train, Y_train)
    coefs2 = {}
    rain_final = {}

    for label, t in thresholds.items():
        q_val = float(label.replace('p', '')) / 100
        mask1 = yfit_train > t
        X1, Y1 = X_train[mask1], Y_train[mask1]
        if X1.empty or Y1.empty:
            continue
        coef1 = quantile_regression(X1, Y1, q_val)
        rain1 = reconstruct_rain(X1, coef1)
        mask2 = rain1 > 0
        X2, Y2 = X1[mask2], Y1[mask2]
        if X2.empty or Y2.empty:
            continue
        coef2 = quantile_regression(X2, Y2, q_val)
        coefs2[label] = coef2
        rain_test = reconstruct_rain(X_test, coef2)
        rain_final[label] = np.maximum(rain_test, 0)

    return coefs2, rain_final

# === CONFIG ===

pca_folder = r'D:\D\Ruvision\GFS\Realtime GFS\Pickle'
obs_excel = pd.read_excel(
    'C:/Users/Angshudeep Majumdar/Documents/Angshudeep Lappy/Ruvision/IMD_2015-2024_Daily Data_0.25 resolution Rain_GFS_LD_1.xlsx'
)
output_base = r'D:\D\Ruvision\GFS\Realtime GFS\Pickle'

thresholds = {'80p': 0.2, '85p': 0.15, '90p': 0.1, '95p': 0.05, '99p': 0.01}

# === MAIN LOOP ===

pattern = r'transformed_dataX_LD_1_(.+)\.pkl'

for pkl in glob.glob(os.path.join(pca_folder, '*.pkl')):
    match = re.search(pattern, os.path.basename(pkl))
    if not match:
        print(f"Skipping file (pattern not matched): {pkl}")
        continue

    var_comb = match.group(1)
    out_dir = os.path.join(output_base, var_comb)
    os.makedirs(out_dir, exist_ok=True)

logger.info("Load data")
    # Load data
    X, Y_df, X_full = load_data(pkl, obs_excel)

    if 'Prec_23.0_72.5' not in Y_df.columns:
        print(f"Column 'Prec_23.0_72.5' not found for {var_comb}. Skipping.")
        continue

    Y = Y_df['Prec_23.0_72.5']

logger.info("Split for training")
    # Split for training
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

logger.info("Train CQM")
    # Train CQM
    coefs2, _ = cqm_pipeline(X_train, Y_train, X_test, thresholds)

    # === FUTURE DATA ===
    X_future = X_full.iloc[len(Y):]
    if X_future.empty:
        print(f"No future data for {var_comb}. Skipping.")
        continue

    last_y_date = Y.index[-1]
    future_dates = pd.date_range(start=last_y_date + pd.Timedelta(days=1), periods=len(X_future), freq='D')
    X_future.index = future_dates

logger.info("Predict for future using trained coefs")
    # Predict for future using trained coefs
    rain_future = {}
    for label, coefs in coefs2.items():
        rain_pred = reconstruct_rain(X_future, coefs)
        rain_future[label] = np.maximum(rain_pred, 0)

logger.info("Extract JJAS 2025")
    # Extract JJAS 2025
    jjas_start_2025 = pd.Timestamp('2025-06-01')
    jjas_end_2025 = pd.Timestamp('2025-09-30')
    mask_jjas_2025 = (X_future.index >= jjas_start_2025) & (X_future.index <= jjas_end_2025)

    rain_future_jjas_2025 = {
        k: pd.Series(v, index=X_future.index)[mask_jjas_2025]
        for k, v in rain_future.items()
    }

    # === CREATE DATAFRAME OF FUTURE PREDICTIONS FOR JJAS 2025 ===
    df_jjas_2025 = pd.DataFrame({
        q_label: series
        for q_label, series in rain_future_jjas_2025.items()
    })

logger.info("Save to Excel (remove 00:00:00 timestamp)")
    # Save to Excel (remove 00:00:00 timestamp)
    df_jjas_2025.index = df_jjas_2025.index.date
    csv_path = os.path.join(out_dir, f'CQM_QuantileForecast_{var_comb}_2025JJAS_LD_1.xlsx')
    df_jjas_2025.to_excel(csv_path, index_label="Date")
    print(f"Forecast DataFrame (with GFS) saved to: {csv_path}")

    # === BAR PLOT: LAST 3 DAYS OF JJAS 2025 for 80th quantile only ===
    q_label = '80p'
    if q_label not in rain_future_jjas_2025:
        print(f"Missing {q_label} for plotting {var_comb}, skipping plot.")
        continue

    pred_series = rain_future_jjas_2025[q_label]
    if pred_series.empty:
        print(f"{q_label} series is empty for {var_comb}, skipping plot.")
        continue

    last_day = pred_series.index[-1:]
    pred_series_1st_day_lead = pred_series.loc[last_day]
pred_series_1st_day_lead


logger.info("In[173]:")
# In[173]:


logger.info("today = pd.Timestamp.today().normalize()- pd.Timedelta(days=1)")
#today = pd.Timestamp.today().normalize()- pd.Timedelta(days=1)
today = pd.Timestamp.today().normalize()


logger.info("In[174]:")
# In[174]:


today


logger.info("In[175]:")
# In[175]:


today_str = today.strftime('%b %d')  # Format as "2025-07-24"


logger.info("In[176]:")
# In[176]:


today_str


logger.info("In[178]:")
# In[178]:


import pandas as pd
import matplotlib.pyplot as plt

# === Combine the three series (first row only if needed) ===
lead_series = pd.concat([
    pred_series_1st_day_lead.iloc[0:1],
    pred_series_2nd_day_lead.iloc[0:1],
    pred_series_3rd_lead.iloc[0:1]
])

# === Plotting ===
fig, ax = plt.subplots(figsize=(15, 6))
bars = ax.bar(lead_series.index, lead_series.values, color='blue', width=0.4)

logger.info("Add value labels")
# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.8, f'{height:.1f}',
            ha='center', va='bottom', fontsize=10)

logger.info("Set title with formatted date")
# Set title with formatted date
ax.set_title(f"Daily based 3 days Rainfall Forecast using Statistical Downscaling model_{today_str} Initialization_Ahmedabad_2025", fontsize=14)
ax.set_ylabel("Rainfall (mm)")
ax.set_xlabel("Forecast Date")
ax.set_xticks(lead_series.index)
ax.set_xticklabels([d.strftime('%d-%b') for d in lead_series.index], rotation=45)

logger.info("Location info")
# Location info
lat_range = (22.5, 23.5)
lon_range = (72, 73)
fig.suptitle(f'Location: Lat {lat_range[0]}°–{lat_range[1]}°, Lon {lon_range[0]}°–{lon_range[1]}°', fontsize=11, y=1.02)

logger.info("Save with formatted date")
# Save with formatted date
fig.tight_layout()
save_path = fr"D:\D\Ruvision\GFS\Realtime GFS\Pickle\U1000_V1000_PREC\forecast_3 Lead Days_{today_str}_Initialisation.png"
fig.savefig(save_path)
plt.close(fig)
print(f"Bar plot saved to: {save_path}")


logger.info("In[ ]:")
# In[ ]:





logger.info("In[ ]:")
# In[ ]:
# === End: Realtime Results Generation_Final.ipynb ===


