import scrapy
import pandas as pd
from scrapy.http import Request
from pymongo import MongoClient
from time import sleep
from datetime import datetime

def getCollection(db, col):
    """Retruns the data for a given database and collection

    :param db: Database that should be returned
    :type db: String
    :param col: Collection that should be returned
    :type col: String
    :return: Data from the given collection
    :rtype: pandas dataframe
    """
    conn = MongoClient("localhost", 27017)
    collobj = conn[db][col]
    collection = pd.DataFrame(list(collobj.find({})))
    return collection.copy()


def insertCollection(db, col, result, drop=False):
    """Inserts the given data into the specified databases collection

    :param drop: To drop the database before inserting or not
    :param db: Database where the collection is
    :type db: string
    :param col: Collection where the data should be stored
    :type col: string
    :param result: Data that should be stored
    :type result: Pandas Dataframe
    """
    result = result.to_dict("records")
    conn = MongoClient("localhost", 27017)
    connObj = conn[db][col]
    if drop:
        connObj.drop()
    connObj.insert_many(result)


class ResumeSpider(scrapy.Spider):
    """
    resumeCrawler_Urls
    - Search urls are created using generateUrls.py using keywords and location details
    resumeCrawler_crawledUrls
    - Search urls that have been crawled for resume urls
    resumeCrawler_resumeUrls
    - This collection has the urls collected from the search urls, for different resumes
    """

    name = "resume"
    crawledUrls = set()
    try:
        crawledUrls = set(getCollection(db="thesisDb_Crawlers", col="resumeCrawler_crawledUrls")['urls'])
    except Exception:
        print("Running first time, no records present!")

    dbUrls = set(getCollection(db="thesisDb_Crawlers", col="resumeCrawler_Urls")['urls'][:10000])
    newUrls = list(dbUrls - crawledUrls)

    def start_requests(self):
        crawlUrls = len(ResumeSpider.newUrls) - 1
        if crawlUrls > 10000:
            crawlUrls = 10000

        for link in ResumeSpider.newUrls[:crawlUrls]:
            urls = pd.DataFrame({"urls": [link], "timestamp": [str(datetime.now())]})
            insertCollection(db="thesisDb_Crawlers", col="resumeCrawler_crawledUrls", result=urls)
            yield Request(link, meta={"mainLink": link})

    def parse(self, response):
        # URL CSS Selector
        url = response.css('div.snippetPadding a::attr(href)').getall()
        url = ["https://www.postjobfree.com" + link for link in url]
        mainLink = []
        currentLink = response.meta["mainLink"]
        timeOfCrawl = datetime.now()
        time = []
        for a in range(len(url)):
            time.append(str(timeOfCrawl))
            mainLink.append(currentLink)
        urls = pd.DataFrame({"urls": url, "timestamp": timeOfCrawl, "mainLink": mainLink})
        insertCollection(db="thesisDb_Crawlers", col="resumeCrawler_resumeUrls", result=urls)
        sleep(2)
