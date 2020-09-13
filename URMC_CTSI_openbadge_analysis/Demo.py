import Dynamic_Network_Graph_Exploration_py3 as dynamic
import Data_Cleaning as dc
import member_to_member_function as mf


'''
So far the main method only includes dynamic network graph functions

Heatmap_functions and Member_Distribution functions to be added 
'''

def main():
    SELECTED_BEACON = 12,
    time_zone = 'US/Eastern'
    log_version = '2.0'
    time_bins_size = '1min'
    members_metadata_filename = "Member-2019-05-28.csv"
    beacons_metadata_filename = "location table.xlsx"
    attendees_metadata_filename = "Badge assignments_Attendees_2019.xlsx"
    data_dir = "../proximity_2019-06-01/"
    
    tmp_m2ms,tmp_m2bs,attendees_metadata,members_metadata= dc.DataCleaning(SELECTED_BEACON,
                                        time_zone,log_version,time_bins_size,
                                        members_metadata_filename,
                                        beacons_metadata_filename,
                                        attendees_metadata_filename,
                                        data_dir)  
    
    
    
    dynamic.NetworkGraphBasicExample('2019-06-01 10:00','2019-06-01 11:20',tmp_m2ms)
    '''
    This function is the code given in the Open Badge Analysis Library 
    '''
    
    dynamic.LunchTimeAnalysis(tmp_m2ms)
    '''
    Find interactive groups during lunch time
    All the parameters in this function are preset 
    '''
    
    dynamic.BreakoutSessionAnalysis(tmp_m2ms)
    '''
    Find interactive groups during breakout session
    All the parameters in this function are preset 
    '''
    
    dynamic.InteractionNetworkGraph(time_interval_start_h=9,time_interval_start_m=50,
                            time_interval_end_h=11,time_interval_end_m=20,
                            interval=2,t_count_threshold = 2,tmp_m2ms=tmp_m2ms)
    '''
    Find interactive groups during any given time with any given parameters 
    The values of the parameters are manually determined by the activities 
    engaged in the time frame given
    '''
    
    # choose a time frame that you are inerested 
    time_slice = slice('2019-06-01 10:00', '2019-06-01 10:00')
    breakout1 = slice('2019-06-01 09:50', '2019-06-01 10:39')
    breakout2 = slice('2019-06-01 10:40', '2019-06-01 11:30')         
    breakout3 = slice('2019-06-01 13:10', '2019-06-01 13:59')   
    breakout4 = slice('2019-06-01 14:00', '2019-06-01 14:50')
    lunch = slice('2019-06-01 11:40','2019-06-01 13:00')
    whole_session = slice('2019-06-01 9:05','2019-06-01 14:50')
    time_period = whole_session
    
    # the function will return heatmaps and the p-value 
    mf.run_all(attendees_metadata,members_metadata,tmp_m2ms,time_period)

if __name__ == "__main__":
    main()
