#create csv file of time vs segment ids with availability 

#imports 
import pandas as pd

timeDict = {}
availDict = {}

#get segment id list 
midpointDf = pd.read_csv("sensor_graph/sfpark_segment_midpoints.csv")
segmentIdList = midpointDf['segmentId'].tolist()

#get dataset 
data = pd.read_csv("sensor_graph/sfpark_filtered_136_247_486taxis.csv", sep=";")

#get avail data at each time for segment id 
for id in segmentIdList:
    timeList = []
    availList = []
    selectedData = data.loc[data['segmentid'] == id]
    print("Current id is:", id)
    for index, row in selectedData.iterrows():
        currentTime = row['timestamp']
        currentAvail = row['capacity'] - row['occupied']
        timeList.append(currentTime)
        availList.append(currentAvail)
    timeDict[id] = timeList
    availDict[id] = availList

#remove segments with incomplete data 
maxId = segmentIdList[0]
for id in segmentIdList[1:]:
    max_duration = len(timeDict[maxId])
    currentDuration = len(timeDict[id])
    if currentDuration < max_duration:
        timeDict.pop(id)
        availDict.pop(id)
    elif currentDuration > max_duration:
        timeDict.pop(maxId)
        availDict.pop(maxId)
        maxId = id

#create csv file 
segmentIds = timeDict.keys()

availDf1 = pd.DataFrame()
availDf1['timestamp'] = timeDict[maxId]

colns = []
for s in segmentIds:
    colns.append(pd.Series(availDict[s], name=s))
availDf2 = pd.concat(colns, names=segmentIds, axis=1)
availDf2 = availDf2.reset_index(drop=True)

availDf = pd.concat([availDf1, availDf2], axis=1)

#save csv file 
availDf.to_csv('sfpark_timestamp_availability_data.csv', index=False)