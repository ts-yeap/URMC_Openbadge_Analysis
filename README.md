# URMC Netowrking Rhythm Badage Analysis
URMC Openbadge Analysis for "Unmeeting" Meeting Data using Python 3

demo how to use 
what to run 
how to get requirement 
what are each file are for 
where is output 

## Authors
* Yumeng Xi
* Haizhu Yang
* Ziyu Song
* Tianyou Xiao


## Set up Python 2 environment on Windows:
https://ipython.readthedocs.io/en/stable/install/kernel_install.html

## Openbadge analysis library
***Download Microsoft Visual C++ Building Tools***
```
https://visualstudio.microsoft.com/downloads/
```
***install ob lib***
```
pip install -e /directory/to/openbadge-analysis --upgrade
```
***install other libraries***
```
pip install -r requirements.txt
```
All code for this project can be found in the [URMC_CTSI_openbadge_analysis]
(https://github.com/yumeng-xi/URMC_Openbadge_Analysis/tree/master/URMC_CTSI_openbadge_analysis) folder.


## Table of Content

### proximity_2019-06-01
```
Data folder.
Please place your data files here, or you can place it somewhere else as long as you modify the data input derectory in 
"./URMC_CTSI_openbadge_analysis/Data_Cleaning.py". 
```

### URMC_CTSI_openbadge_analysis
```
Python 3 file forder.
Please see the below "Codebook" for more details. 
```

### README.md
```
Read me. 
```

### requirement.txt
```
Required. 
```


## Codebook
### Preprocessing (Preprocessing.py)
```
Preprocess input txt data files. 
```

### Data_Clearning (Data_cleaning.py)
```
Clean data and manipulate data into the structures we need for futher analysis. 
```

### Signal Strength Distribution (signal_strength_distribution.py):
```
We generated histograms for the distribution of signal strength with count as frequency. We also created dynamic signal strength 
frequency change in a video format which is available through a share-only link on Youtube.

Video demo link: 
Lunch break: https://youtu.be/4X9Xs9C4Gqw 
Breakout Session 1: https://youtu.be/7wvly0PomRs 
Breakout Session 2: https://youtu.be/f8ifvLcI7EE 
Breakout Session 3: https://youtu.be/VeF3HaHZXjo 
Breakout Session 4: https://youtu.be/eSf5lXNmwaw 
```

### Dynamic Network Graph (Dynamic_Network_Graph_Exploration_py3.py)
#### Network Graph Basic Example
```
This function will draw the dynamic network graph with average signal strengths among members and their relative average location 
in the graph. 

```
#### Lunch Time Analysis
```
This analysis focuses on lunchtime and will draw multiple graphs of the lunchtime every 5-minute interval. The signal strength 
threshold, count threshold are customized. Parameters can be modified, but mostly only need to run the section.
```
#### Breakout Session Analysis
```
This analysis focuses on breakout sessions and will draw multiple graphs of the breakout session every 2-minute interval. The 
signal strength threshold, count threshold are customized. Parameters can be modified, but mostly one only need to run the section 
to see the results. 
```
#### Interaction Network Graph
```
This tool helps find the interaction between members in a certain amount of time with designated parameters. The parameters such 
as signal strength using the distribution of signal strength. After choosing a threshold for signal strength, the program analyzes 
the distribution of frequencies of signal strength and it will take the frequency of the peak-2. (-2 for leave some room for 
fluctuation).
```
#### Interaction Network Graph with Attendees' Background Information
```
This tool addes attendees' background information to the previous Interaction Network Graph to better present how the attendees' 
backgrounds influence their interactions with other attendees.
```
### Member Interaction Distribution (member_to_member_function.py)
```
At the beginning of the script, it cleans out the badge’s information that was not worn by attendees. This script generates 
heatmaps about the member to member interaction given any time period. For the different time periods, the user needs to pick a 
threshold for this session based on the signal strength distribution histograms. Part of the script also generates heatmap 
interaction distribution between members with the same background fields and members with different background fields. The script 
also generates p-values for the statistical hypothesis test. 
```

### Room Assignment
```
By finding the closest beacon to each member during each predefined time period (in our case, half breakout session), we can 
“assign” them the room they probably stayed in during that session, as we already knew the locations of each beacon. We also 
verify our outcome in the dynamic member network graph by giving each node different colors based on the room assignment 
information. 
```

### Demo (Demo.py)
```
This python file gives examples of how to use each function. 
```
