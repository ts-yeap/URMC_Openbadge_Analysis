# loading necessary libraries

from Preprocessing import *

import os, sys
import logging
import json
import gzip

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import math
from matplotlib import colors
import matplotlib.patches as mpatches
import matplotlib.cm as cm

from collections import Counter


'''
Settings (directory, time zone, etc.)
'''

time_zone = 'US/Eastern'
log_version = '2.0'
time_bins_size = '1min'

proximity_data_filenames = []

# get the strings of all the filenames for file reading
for i in range(1, 18):
    if i < 10:
        filename = 'CTSIserver{:02d}_proximity_2019-06-01.txt'.format(i)
    else:
        filename = 'CTSIserver{}_proximity_2019-06-01.txt'.format(i)
        
    proximity_data_filenames.append(filename)

# data directory preset
members_metadata_filename = "Member-2019-05-28.csv"
beacons_metadata_filename = "location table.xlsx"
data_dir = "./proximity_2019-06-01/"


'''
function definition
'''
def time_location_plot(time_locations, ax=None, cmap=None, freq='30min', datetime_format='%H:%M'):
    """Plots the location of individuals/groups of individuals as a function of time.
    
    Parameters
    ----------
    time_locations : pd.Series
        The locations, indexed by 'datetime' and another index (e.g. member, group, ...).
    
    ax : matplotlib axes
        The axes on which to plot.
    
    cmap : matplotlib.colors.Colormap or str
        The colormap used by matplotlib.
    
    freq : str
        The frequency of ticks on the x-axis.  Defaults to '30min'.
    
    datetime_format : str
        The way time is formatted with `strftime` on the x-axis.
        See https://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior
        for more information.
    """

    # Pivot the dataframe to have datetime vs member
    pivoted = time_locations.unstack(0)

    # The list of locations
    locations = list(set(time_locations))
    N = len(locations)

    # Select a colormap (either passed, string or default)
    if cmap is None:
        cmap = plt.get_cmap('tab10')  # Default colormap
    elif type(cmap) is str:
        cmap = plt.get_cmap(cmap)

    if ax is None:
        ax = plt.gca()  # Use current axes if none were provided

    # Construct a colormap based on `cmap` that matches the number of different locations,
    # and displays `None` as white
    cmap = mpl.colors.ListedColormap([(1.0, 1.0, 1.0)] + list(cmap.colors), N=N+1)

    # Map locations to integers, to be used by `pcolor`
    mapping = {loc: i+1 for (i, loc) in enumerate(locations)}
    pivoted = pivoted.applymap(lambda x: mapping.get(x, 0))

    # Create the time-location diagram
    coll = ax.pcolormesh(pivoted, cmap=cmap)

    # Vertical axis, with the people
    ax.set_yticks(np.arange(0.5, len(pivoted.index), 1))
    ax.set_yticklabels(pivoted.index)

    # Horizontal axis, with the dates
    # xlabels = pd.date_range(pivoted.columns[0], pivoted.columns[-1], freq=freq)
    # ax.set_xticks([pivoted.columns.get_loc(idx, method='nearest') for idx in xlabels.strftime(datetime_format)])
    # ax.tick_params(axis='x',labelsize=20)
    # ax.set_xticklabels(xlabels, rotation=45, ha='right')

    # Add a colorbar, with one tick per location
    cbar = ax.figure.colorbar(coll, use_gridspec=False,
                              anchor=(-.2, 0.5), aspect=30,
                              boundaries=np.arange(0, N+1)+.5)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_ticks(np.arange(1., N+1, 1.))
    cbar.set_ticklabels(locations)

    plt.savefig('./output/time_loc.jpg')
    print('time location graph saved.')


def draw_graph(G, graph_layout='shell',
               node_size=200, node_color='blue', node_alpha=0.3,
               node_text_size=6,
               edge_color='blue', edge_alpha=0.3, edge_tickness=1,
               edge_text_pos=0.3,
               text_font='sans-serif', draw_name=True):

    # these are different layouts for the network you may try
    # shell seems to work best
    if graph_layout == 'spring':
        graph_pos=nx.spring_layout(G)
    elif graph_layout == 'spectral':
        graph_pos=nx.spectral_layout(G)
    elif graph_layout == 'random':
        graph_pos=nx.random_layout(G)
    else:
        graph_pos=nx.shell_layout(G)
        
    
    # draw graph
    nx.draw_networkx_nodes(G,graph_pos,node_size=node_size, 
                           alpha=node_alpha, node_color=node_color, cmap=plt.get_cmap('tab10'))
    nx.draw_networkx_edges(G,graph_pos,width=edge_tickness,
                           alpha=edge_alpha,edge_color=edge_color)

    if draw_name:
        nx.draw_networkx_labels(G, graph_pos,font_size=node_text_size,
                                font_family=text_font)


    plt.savefig('./output/m2m_graph.jpg')
    print("Member-to-member graph saved.")


# a modified draw_graph method with colors based on room assignments
def draw_graph_colored(G, graph_layout='shell',
               node_size=200, node_color='blue', node_alpha=0.5,
               node_text_size=6,
               edge_color='blue', edge_alpha=0.3, edge_tickness=1,
               edge_text_pos=0.3,
               text_font='sans-serif', draw_name=True):

    # these are different layouts for the network you may try
    # shell seems to work best
    if graph_layout == 'spring':
        graph_pos=nx.spring_layout(G)
    elif graph_layout == 'spectral':
        graph_pos=nx.spectral_layout(G)
    elif graph_layout == 'random':
        graph_pos=nx.random_layout(G)
    else:
        graph_pos=nx.shell_layout(G)        
    
    cMap=plt.get_cmap('jet')
    
    
    tmp_nc = []
    tmp_nr = []
    for member in nx.get_node_attributes(G, 'region'):
        tmp_nc.append(nx.get_node_attributes(G, 'color')[member])
        tmp_nr.append(nx.get_node_attributes(G, 'region')[member])
    
    ncolor = cMap(tmp_nc)
    
    # draw graph
    nx.draw_networkx_nodes(G,graph_pos,node_size=node_size,
                           alpha=node_alpha, node_color=ncolor,cmap=plt.get_cmap('jet'))
    nx.draw_networkx_edges(G,graph_pos,width=edge_tickness,
                           alpha=edge_alpha,edge_color=edge_color)

    if draw_name:
        nx.draw_networkx_labels(G, graph_pos,font_size=node_text_size,
                                font_family=text_font)

    
    plt.savefig('./output/colored_m2m_graph.jpg')
    print("Colored dynamic m2m graph saved.")

# generate a time slice starting with the given time with length "duration"
def generate_time_slices_by_duration(start_h, start_m, duration):
    start = '2019-06-01 {:02}:{:02}'.format(start_h, start_m)
    duration_h = (int)(duration/60)
    duration_m = duration % 60
    if duration_m + start_m > 60:
        duration_h += 1
        duration_m = (duration_m+start_m) % 60
        end = '2019-06-01 {:02}:{:02}'.format(start_h+duration_h, duration_m-1)
        return start_h+duration_h, duration_m-1, slice(start, end)
    else:
        end = '2019-06-01 {:02}:{:02}'.format(start_h+duration_h, duration_m+start_m-1)
        return start_h+duration_h, duration_m+start_m-1, slice(start, end)

    
# create time slices between start time and end time by every "interval" minute(s)
def generate_time_slices_by_interval(start_h, start_m, end_h, end_m, interval=2):
    
    time_slices = []
    
    start = '2019-06-01 {:02}:{:02}'.format(start_h, start_m)
    duration = (end_h - start_h) * 60 + (end_m - start_m)
    
    for i in range(duration/interval+1):
        
        if start_h > end_h:
            break
        elif start_h == end_h:
            if start_m > end_m:
                break
            
        tmp_h = start_h
        tmp_m = start_m
        
        if tmp_m < 60-interval+1:
            tmp_m += interval-1
        else:
            tmp_h += 1
            if interval > 1:
                tmp_m = tmp_m - 60 + interval-1
            else:
                tmp_m = tmp_m - 60 + interval
            
        tmp_time = '2019-06-01 {:02}:{:02}'.format(tmp_h, tmp_m)
        
        time_slices.append(slice(start, tmp_time))
        
        start_h = tmp_h
        start_m = tmp_m
        
        if start_m == 59:
            start_h += 1
            start_m = 0
        else:
            start_m += 1
        
        start = '2019-06-01 {:02}:{:02}'.format(start_h, start_m)
    return time_slices

    # Filter data from specific time period


'''
Preprocessing
'''

# read beacon and member metadata
members_metadata = pd.read_csv(data_dir+members_metadata_filename)
members_metadata.dropna()
beacons_metadata = pd.read_excel(data_dir+beacons_metadata_filename, sheet_name='Sheet1')

# read member background data, merge to the original member metadata to keep consistent
background = pd.read_excel(data_dir+'Badge assignments_Attendees_2019.xlsx')
cleaned_members = members_metadata.merge(background, how='inner')

# separate out the member key strings of people who actually attended the meeting and those who did not
attendees_key = set(cleaned_members['member'])
all_people_key = set(members_metadata['member'])
attendees_id = set(cleaned_members['id'])
all_people_id = set(members_metadata['id'])

non_attendees_key = all_people_key-attendees_key
non_attendees_id = all_people_id-attendees_id

print('data loading done')


'''
beacon metadata
'''
idmaps = []

for proximity_data_filename in proximity_data_filenames:
    with open(os.path.join(data_dir, proximity_data_filename), 'r') as f:
        idmaps.append(id_to_member_mapping(f, time_bins_size, tz=time_zone))


m2badges = []

for proximity_data_filename in proximity_data_filenames:
    with open(os.path.join(data_dir, proximity_data_filename), 'r') as f:
        m2badges.append(member_to_badge_proximity(f, time_bins_size, tz=time_zone))


cleaned_m2badges = []

for m2badge in m2badges:
    drop_list = []
    tmp = m2badge.reset_index()

    for index, row in tmp.iterrows():
        if row['member'] in non_attendees_key:
            drop_list.append(index)
        else:
            if row['observed_id'] in non_attendees_id:
                drop_list.append(index)
    tmp = tmp.drop(drop_list)
    cleaned_m2badges.append(tmp)


print('beacon metadata loading done')




'''
data transformation
'''

# Member to member
m2ms = []
for (m2badge, idmap) in zip(cleaned_m2badges, idmaps):
    m2ms.append(member_to_member_proximity(m2badge, idmap))
    
tmp_m2ms = m2ms[0]
for i in range(1, len(m2ms)):
    tmp_m2ms = pd.concat([tmp_m2ms, m2ms[i]])

# Member to location beacon
m2bs = []
for m2badge in cleaned_m2badges:
    m2bs.append(member_to_beacon_proximity(m2badge, beacons_metadata.set_index('id')['beacon']))
    
tmp_m2bs = m2bs[0]
for i in range(1, len(m2bs)):
    tmp_m2bs = pd.concat([tmp_m2bs, m2bs[i]])


m5cb = tmp_m2bs.reset_index().groupby(['datetime', 'member'])[['rssi', 'beacon']] \
        .apply(lambda x: x.nlargest(20, columns=['rssi']) \
        .reset_index(drop=True)[['beacon']]).unstack()['beacon'].fillna(-1).astype(int)


print('data transformation done')


'''
Example 1: Network Graph
'''

tmp_m2ms_sorted = tmp_m2ms.sort_index(0,0)

# create time slices of four breakout sessions for future usage
breakout1 = slice('2019-06-01 09:50', '2019-06-01 10:39')
breakout2 = slice('2019-06-01 10:40', '2019-06-01 11:30')
breakout3 = slice('2019-06-01 13:10', '2019-06-01 13:59')
breakout4 = slice('2019-06-01 14:00', '2019-06-01 14:50')

# Filter data from specific time period

m2m_breakout = tmp_m2ms_sorted.loc[breakout1]

# keep only instances with strong signal
m2m_filter_rssi = m2m_breakout[m2m_breakout.rssi >= -70].copy()
print("Number of \"strong\" signals: ", len(m2m_filter_rssi))


# Count number of time members were in close proximity
# We name the count column "weight" so that networkx will use it as weight for the spring layout
m2m_edges = m2m_filter_rssi.groupby(['member1', 'member2'])[['rssi_weighted_mean']].count().rename(columns={'rssi_weighted_mean':'weight'})
m2m_edges = m2m_edges[["weight"]].reset_index()

# Keep strongest edges (threshold set manually)
m2m_edges = m2m_edges[m2m_edges.weight > 5]
print("# of edges left: ", len(m2m_edges))

# Create a graph
graph=nx.from_pandas_edgelist(m2m_edges, "member1", "member2", "weight")


# set colors by region
regions = members_metadata["region"].unique()
len_regions = len(regions)
value_map = {}
for i in range(0,len_regions):
    value_map[regions[i]] = i * (1.0/(len_regions-1))


# Add node information. Assign color
for index, row in cleaned_members.iterrows():
    member = row["member"]
    region = row["region"]
    color = value_map.get(region)
    if member in graph.nodes():        
        graph.add_node(member, region=region, color = color)
        
node_colors = [color for member,color in nx.get_node_attributes(graph, 'color').items()]

fig = plt.figure(figsize=(12, 10), dpi=150)
ax = plt.subplot(1,1,1)
ax.set_title('Member-to-member Network')
draw_graph(graph, graph_layout="spring", node_color = node_colors, draw_name=False)



'''
Example 2: Closest Beacon Localization
'''

# update the memebr metadata with new index
members_metadata_m = cleaned_members.set_index('member')
beacons_metadata_b = beacons_metadata.set_index('beacon')

# Take the closest beacon for each member
closest = m5cb[0].rename('beacon')


# Go from closest beacon to location using the metadata
locations = closest.to_frame().join(beacons_metadata_b['location'], on='beacon')

# Join people's region for sorting
locations = locations.join(members_metadata_m['region'])

# Add the region to the index
locations = locations.reset_index().set_index(['datetime', 'region', 'member']).sort_index()

locations = locations['location']
# print(locations.head(10))

# The time slice for displaying the heatmap
time_slice = slice('2019-06-01 07:50', '2019-06-01 14:50')

# Plotting
fig = plt.figure(figsize=(20, 20), dpi=150)
ax = plt.subplot(1,1,1)

time_location_plot(locations.loc[time_slice])



'''
Example 3
'''

# create 8 half-breakout session time slices, saved in variable half_breakout_sessions

# 1st two halfs
total_start_h = 9
total_start_m = 50
half_breakout_sessions = []
for i in range(2):
    total_start_h, total_start_m, ts = generate_time_slices_by_duration(total_start_h, total_start_m, 23)
    total_start_m += 1
    half_breakout_sessions.append(ts)
    
# 2nd two halfs
total_start_m += 4
total_start_h += (int)(total_start_m/60)
total_start_m = total_start_m % 60
for i in range(2):
    total_start_h, total_start_m, ts = generate_time_slices_by_duration(total_start_h, total_start_m, 23)
    total_start_m += 1
    half_breakout_sessions.append(ts)

# 3rd two halfs
total_start_h = 13
total_start_m = 10
for i in range(2):
    total_start_h, total_start_m, ts = generate_time_slices_by_duration(total_start_h, total_start_m, 23)
    total_start_m += 1
    half_breakout_sessions.append(ts)
    
# 4th two halfs
total_start_m += 4
total_start_h += (int)(total_start_m/60)
total_start_m = total_start_m % 60
for i in range(2):
    total_start_h, total_start_m, ts = generate_time_slices_by_duration(total_start_h, total_start_m, 23)
    total_start_m += 1
    half_breakout_sessions.append(ts)


# get the room assignment for each member in each half-breakout session

df_half_breakout = []
for time_slice in half_breakout_sessions:
    df_half_breakout.append(locations.loc[time_slice].reset_index())
    
member_keys = set(pd.DataFrame(locations).reset_index()['member'].unique())

location_dict_half_breakout = []

for i in range(8):
    tmp = df_half_breakout[i]
    tmp_dict = {}
    for j in member_keys:
        tmp_dict.setdefault(j, [])
    for row in tmp.iterrows():
        tmp_key = row[1][2]
        tmp_room = row[1][3]

        tmp_dict[tmp_key].append(tmp_room)
    location_dict_half_breakout.append(tmp_dict)

cleaned_dict = dict()
for i in member_keys:
    cleaned_dict.setdefault(i, [])

for i in range(8):
    tmp_dict = location_dict_half_breakout[i]
    
    for member in list(member_keys):
        member_hist = tmp_dict[member]
        c = Counter(member_hist)
        
        if len(c.most_common(1)) > 0:
            if c.most_common(1)[0][1] < 6:
                continue        
            cleaned_dict[member].append(c.most_common(1)[0][0])


# add "not here"
for member in cleaned_dict.keys():
    if len(cleaned_dict[member]) < 8:
        add = 8 - len(cleaned_dict[member]) 
        for i in range(add):
            cleaned_dict[member].append('Not Here')


# extract room assignments of each half-breakout session
room_assignments = [[] for _ in range(8)]



for row in cleaned_members.iterrows():
    member = row[1][0]
    cleaned_dict.setdefault(member, ['Not Here']*8)
    for i in range(8):
        room_assignments[i].append(cleaned_dict[member][i])
    
member_metadatas = []
    
for i in range(8):
    tmp_member_metadata = cleaned_members.copy()
    tmp_member_metadata['region'] = room_assignments[i]
    member_metadatas.append(tmp_member_metadata.set_index('member'))

# Filter data from specific time period

SELECTED_HALF_BREAKOUT = 0

m2m_half_breakout1 = tmp_m2ms_sorted.loc[half_breakout_sessions[SELECTED_HALF_BREAKOUT]]
# m2m_half_breakout1 = tmp_m2ms_sorted.loc[breakout1]

# keep only instances with strong signal
m2m_filter_rssi1 = m2m_half_breakout1[m2m_half_breakout1.rssi >= -72].copy()


# Count number of time members were in close proximity
# We name the count column "weight" so that networkx will use it as weight for the spring layout
m2m_edges1 = m2m_filter_rssi1.groupby(['member1', 'member2'])[['rssi_weighted_mean']].count().rename(columns={'rssi_weighted_mean':'weight'})
m2m_edges1 = m2m_edges1[["weight"]].reset_index()

# Keep strongest edges (threshold set manually)
m2m_edges1 = m2m_edges1[m2m_edges1.weight > 1]

# Create a graph
graph1=nx.from_pandas_edgelist(m2m_edges1, "member1", "member2", "weight")


# set colors by region
regions1 = member_metadatas[SELECTED_HALF_BREAKOUT]["region"].unique()
len_regions1 = len(regions1)
value_map1 = {}
for i in range(0,len_regions1):
    value_map1[regions1[i]] = i * (1.0/(len_regions1-1))

# Add node information. Assign color
for index, row in member_metadatas[SELECTED_HALF_BREAKOUT].reset_index().iterrows():
    member = row["member"]
    region = row["region"]
    color = value_map1.get(region)
    if member in graph1.nodes():   
        graph1.add_node(member, region=region, color=color)
        
node_colors1 = [color for member,color in nx.get_node_attributes(graph1, 'color').items()]
node_region1 = [region for member, region in nx.get_node_attributes(graph1, 'region').items()]


# get the list of region and color following the order by which Networkx plots node
nc = []
nr = []
for key in list(graph1.nodes()):
    nc.append(nx.get_node_attributes(graph1, 'color')[key])
    nr.append(nx.get_node_attributes(graph1, 'region')[key])


# create a map from color to region for plotting legend
scalar2region = {}
for i in range(len(node_colors1)):
    scalar2region[nc[i]] = nr[i]


fig = plt.figure(figsize=(12, 10), dpi=500)
ax = plt.subplot(1,1,1)
plt.title('First Half Breakout Session')
draw_graph_colored(graph1, graph_layout="spring", node_color = nc, draw_name=True)