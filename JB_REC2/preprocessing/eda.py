from JB_REC2.connections.mongoConnection import getCollection, insertCollection

df = getCollection("thesisDb_Crawlers", "glassdoorJobListings")

df = df[["job.listingId.long",
         "header.jobTitle",
         "header.location",
         "job.description",
         "job.jobSource",
         "map.country",
         "map.employerName",
         "map.location",
         "overview.industry",
         "overview.sector",
         "overview.size",
         "overview.description"]]

for eachCol in df:
    df[eachCol] = df[eachCol].astype('str')

for eachCol in df:
    # print(eachCol)

    # wordcount = len(df[eachCol].str.split().max())

    max1 = df[eachCol].str.len().max()
    # min1 = df[eachCol].str.split().min()
    print("\hline")
    print(f"{eachCol} & string & {max1}\\\\")

maxs = 0
counter = 0
mean = 0
for d in df['job.description'].iteritems():
    strs = str(d[1])
    strs = strs.split()
    print(strs)
    lens = len(strs)
    if lens > maxs:
        maxs = lens
    print(lens)
    print(maxs)
    counter = counter + 1
    mean = mean + lens

mean = mean / counter
print(f"mean: {mean}")



from JB_REC2.connections.mongoConnection import getCollection, insertCollection

df = getCollection("thesisDb_Crawlers", "resumeCrawler_profiles")

df = df[['title', 'location', 'postedOn', 'resume']]

for eachCol in df:
    df[eachCol] = df[eachCol].astype('str')

for eachCol in df:
    # print(eachCol)

    # wordcount = len(df[eachCol].str.split().max())

    max1 = df[eachCol].str.len().max()
    # min1 = df[eachCol].str.split().min()
    print("\hline")
    print(f"{eachCol} & string & {max1}\\\\")

maxs = 0
counter = 0
mean = 0
for d in df['resume'].iteritems():
    strs = str(d[1])
    strs = strs.split()
    print(strs)
    lens = len(strs)
    if lens > maxs:
        maxs = lens
    print(lens)
    print(maxs)
    counter = counter + 1
    mean = mean + lens

mean = mean / counter
print(f"mean: {mean}")
