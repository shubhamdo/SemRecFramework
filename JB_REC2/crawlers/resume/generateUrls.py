import pandas as pd
from resume.resume.spiders.mongoConnection import insertCollection


def resumeUrls():
    keywords = pd.read_csv("H:/Thesis/Input_Data/keywords.csv")
    locations = pd.read_csv("H:/Thesis/Input_Data/locations.csv")

    start_url = []
    for index, row in keywords.iterrows():
        for indexLoc, rowLoc in locations.iterrows():
            # print(row[0], rowLoc[0])
            s = "https://www.postjobfree.com/resumes?q=" + row[0] + "&l=" + rowLoc[0] + "&radius=25"
            start_url.append(s)

    urls = pd.DataFrame({"urls": start_url})
    insertCollection(db="thesisDb_Crawlers", col="resumeCrawler_Urls", result=urls)


def jobsUrls():
    # TODO Add code for jobsURLs
    pass
