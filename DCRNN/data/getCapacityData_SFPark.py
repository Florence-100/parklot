#create csv file of time vs segment ids with capacity 

#imports 
import pandas as pd 

timeDict = {}
capacityDict = {}

#get segment id list 
midpointDf = pd.read_csv("sensor_graph/sfpark_segment_midpoints.csv")
segmentIdList = midpointDf['segmentId'].tolist()

#get dataset 
data = pd.read_csv("sensor_graph/sfpark_filtered_136_247_486taxis.csv", sep=";")

#get capacity data at each time for segment id 
for id in segmentIdList:
    timeList = []
    capacityList = []
    selectedData = data.loc[data['segmentid'] == id]
    print("Current id is:", id)
    for index, row in selectedData.iterrows():
        currentTime = row['timestamp']
        currentCapacity = row['capacity']
        timeList.append(currentTime)
        capacityList.append(currentCapacity)
    timeDict[id] = timeList
    capacityDict[id] = capacityList

#remove segments with incomplete data 
maxId = segmentIdList[0]
for id in segmentIdList[1:]:
    max_duration = len(timeDict[maxId])
    currentDuration = len(timeDict[id])
    if currentDuration < max_duration:
        timeDict.pop(id)
        capacityDict.pop(id)
    elif  currentDuration > max_duration:
        timeDict.pop(maxId)
        capacityDict.pop(maxId)
        maxId = id

#create csv file 
segmentIds = timeDict.keys()

availDf1 = pd.DataFrame()
availDf1['timestamp'] = timeDict[maxId]

colns = []
for s in segmentIds:
    colns.append(pd.Series(capacityDict[s], name=s))
availDf2 = pd.concat(colns, names=segmentIds, axis=1)
availDf2 = availDf2.reset_index(drop=True)

#save csv file 
capacityDf = pd.concat([availDf1, availDf2], axis=1)
capacityDf.to_csv('sfpark_timestamp_capacity_data.csv', index=False)