#create csv file of time vs carpark ids with availability 

#imports 
import pandas as pd

timeDict = {}
availDict = {}

#get segment id list 
carparkIdDf = pd.read_csv("sensor_graph/Nottingham Carpark Locations.csv")
carparkIdList = carparkIdDf['Carpark Id'].tolist()

#get dataset 
data = pd.read_csv("sensor_graph/Nottingham_demo_dataset2_more.csv")

#get avail data at each time for segment id 
for id in carparkIdList:
    timeList = []
    availList = []
    selectedData = data.loc[data['carparkId'] == id]
    print("Current id is:", id)
    for index, row in selectedData.iterrows():
        currentTime = row['timestamp']
        currentAvail = row['availability']
        timeList.append(currentTime)
        availList.append(currentAvail)
    timeDict[id] = timeList
    availDict[id] = availList


#check length of each carpark data 

for key, value in timeDict.items():
    datalen = len(value)
    print(f"Carpark {key} has {datalen} data points.")



#create csv file 

availDf1 = pd.DataFrame()
availDf1['timestamp'] = timeDict[carparkIdList[0]]

colns = []
for s in carparkIdList:
    colns.append(pd.Series(availDict[s], name=s))
availDf2 = pd.concat(colns, names=carparkIdList, axis=1)
availDf2 = availDf2.reset_index(drop=True)

availDf = pd.concat([availDf1, availDf2], axis=1)

#save csv file 
availDf.to_csv('Nottingham_demo_timestamp_availability_data.csv', index=False)

