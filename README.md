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

### README.md
```

```
### requirement.txt
```

```
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
```
Python 3 file forder.
Please see the below "Codebook" for more details. 
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
Python 3 file forder.
Please see the below "Codebook" for more details. 
```
### Demo (Demo.py)
```
This python file gives examples of how to use each function. 
```
