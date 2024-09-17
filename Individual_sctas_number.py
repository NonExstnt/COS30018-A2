import pandas as pd

SCATS_SITES = r"C:\Users\ASUS\OneDrive\Semester 3\Intelligent System\Assignment\Datasets\SCATS Site Listing.csv"
SCATS_SITE_TYPES = r"C:\Users\ASUS\OneDrive\Semester 3\Intelligent System\Assignment\Datasets\SCATS Site Types.csv"
SITE_DATA = r"C:\Users\ASUS\OneDrive\Semester 3\Intelligent System\Assignment\Datasets\SCATS Data October 2006.csv"

scats_sites = pd.read_csv(SCATS_SITES)
scats_site_types = pd.read_csv(SCATS_SITE_TYPES)
scats_site_data = pd.read_csv(SITE_DATA)

import re

def export_scats_site_data(scat_number):

    scats_site_data['Date'] = pd.to_datetime(scats_site_data['Date'], format='%m/%d/%Y')
    start_date = '2006-10-01'
    end_date = '2006-10-31'

    selected_scat_data = scats_site_data[(scats_site_data['SCATS Number'] == scat_number) &
                                         (scats_site_data['Date'] >= start_date) &
                                         (scats_site_data['Date'] <= end_date)]

    Time_column = [col for col in selected_scat_data.columns if re.match(r'^\d{1,2}:\d{2}$', col)]

    Modified_data = selected_scat_data.melt(id_vars=['Location', 'Date'], value_vars=Time_column, var_name='Time', value_name='Vehicle Count')

    Modified_data = Modified_data.sort_values(by=['Date', 'Time']).reset_index(drop=True)

    data_subset = Modified_data[['Location', 'Date', 'Time', 'Vehicle Count']]

    #print(data_subset.head().to_string(index=False))

    data_subset.to_csv(f'individual_data/Scat_number_{scat_number}.csv', index=False)

locations = scats_site_data.drop_duplicates(subset=["SCATS Number"])

unique_scats_list = locations['SCATS Number'].unique()

for i in range(40):
    scat_number = unique_scats_list[i]

    export_scats_site_data(scat_number)