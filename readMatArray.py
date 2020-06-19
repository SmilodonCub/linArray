# -*- coding: utf-8 -*-
"""
Read Linear Array .mat file
"""

# %% environmental
#load methods
from scipy.io import loadmat
import pandas as pd

# %%
#load .mat into python environment
linArrayPath = "/home/bonzilla/Documents/smilodonCub_fossils/MATLAB/linArrayWorkspace.mat"
linArrayMat = loadmat( linArrayPath )

# %% define helper functions

delayedSaccade_dat = linArrayMat[ 'delayedSaccIntervals' ]
delayedSaccade_DF = pd.DataFrame( data = delayedSaccade_dat, columns = [ 'start','end' ] )

spike09_dat = linArrayMat[ 'SPK09a' ]

def getTrialTimestamps( array_DF, trialDat_DF, task, channel ):
    """
    generates a list for each trial within a list for a specific task.
    array_DF: pandas dataframe with array data
    task: string for task batched data intervals
    channel: string for a plexon channel (e.g. SPK or FP)
    returns a list of lists with the events for a given trial interval
    """
    
    channelDat = array_DF[ channel ]
    channelDat_DF = pd.DataFrame( data = channelDat, 
                                 columns = [ 'timestamps' ] )
    numTrials = len( trialDat_DF[ 'start' ] )
    timeStampLists = pd.Series([])

    for aTrial in range( numTrials ):
        startTime = trialDat_DF[ 'start' ][ aTrial ]
        endTime = trialDat_DF[ 'end' ][ aTrial ]
        trialTimeStamps_IDX = channelDat_DF[ 'timestamps' ].between( startTime, endTime )
        trialTimeStamps = channelDat_DF[ trialTimeStamps_IDX ]
        timeStampLists[ aTrial ] = trialTimeStamps   
    return timeStampLists

#alist = getTrialTimestamps( linArrayMat, 'delayedSaccIntervals', 'SPK09a')
# %% 

# a dataframe that holds columns names for all channels with spike data    
allKeys = pd.DataFrame( linArrayMat.keys(), columns = [ 'keys' ] )
spikeKeys_IDX = allKeys['keys'].str.startswith( 'SPK' )
spikeKeys = allKeys[ spikeKeys_IDX ]
print( len( spikeKeys ) )

#get spikes for each trial for each channel

def getMultiChannelTimestamps( array_DF, channels_df, task ):
    """
    given data and a specific task, returns a dataframe with a row for each 
    trial a feature for start and end timestamps and features for each channel 
    queried.
    array_DF: pandas dataframe with array data
    channels_DF: pandas dataframe with a feature of strings, one for each 
    plexon channel to be collected on
    task: string for task batched data intervals
    """
    trialDat = array_DF[ task ]
    trialDat_DF = pd.DataFrame( data = trialDat, 
                               columns = [ 'start', 'end' ] )
    #for each item in channel_df:
    for aChannel in range( len( channels_df ) ):    
        #getTrilTimestamps
        channel = channels_df[ 'keys' ].values[ aChannel ]
        channelTrialDat = getTrialTimestamps( array_DF, trialDat_DF, task, channel )
        #print( type( channelTrialDat ) )
        trialDat_DF[ channel ] = channelTrialDat
        #add as a new feature to trialDat_DF
    return trialDat_DF
    
df = getMultiChannelTimestamps( linArrayMat, spikeKeys, 'delayedSaccIntervals' )

# %% add event marker

def addEventMarker( array_DF, trialDat_DF, eventmarker, task ):
    """
    takes in array dataframe and finds the timestamps for a specified
    eventmarker for each trial to be added as a feature column to a 
    dataframe of trial data (i.e. output from getMultiChannelTimestamps())
    array_DF: pandas dataframe generated from a matlab structure summarizing array data
    trialDat_DF: pandas dataframe with data by trial (output from getMultiChannetTimestamps())
    eventmarker: int, specifies which eventmarker to parse from array_DF
    """
    #make a string for the strobed event column label
    numZeros = 12 - 7 - len( str( eventmarker ) )
    strobedLabel = 'Strobed' + '0'*numZeros + str( eventmarker )

    #check that the eventmarker is present
    colNames = pd.DataFrame( array_DF.keys(), columns = [ 'keys' ] )
    if colNames[ 'keys' ].str.contains( strobedLabel ).any():
        taskEventMarkers = getTrialTimestamps( array_DF, trialDat_DF, \
                                              task, strobedLabel )
    else:
        raise Exception( 'Sorry, eventmarker not present\nbe sure to enter \
                        only the integer value of the event\nex: CORRECT: 14 \
                        INCORRECT: 00014, Strobed00014')
    trialDat_DF[ strobedLabel ] = taskEventMarkers    
    return trialDat_DF

df = addEventMarker( linArrayMat, df, 1000, 'delayedSaccIntervals' )


