import scrapy
from bs4 import BeautifulSoup
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


class ResumeDetailSpider(scrapy.Spider):
    """
    resumeCrawler_resumeUrls
    - This collection has the urls collected from the search urls, for different resumes
    resumeCrawler_resumeCrawledUrls
    - Resume urls that have been crawled for resume details
    resumeCrawler_profiles
    - Resume urls are crawled for the details like location, title, resume text, and link of resume
    """

    name = "resumedetail"
    crawledUrls = set()
    dbUrls = set()
    currentUrl = ""
    try:
        crawledUrls = set(getCollection(db="thesisDb_Crawlers", col="resumeCrawler_resumeCrawledUrls")['urls'])
    except Exception:
        print("Running first time, no records are crawled at present!")

    try:
        dbUrls = set(getCollection(db="thesisDb_Crawlers", col="resumeCrawler_resumeUrls")['urls'][:10000])
    except Exception:
        print("Running first time, no records present for resumes!")

    newUrls = list(dbUrls - crawledUrls)

    def start_requests(self):
        crawlUrls = len(ResumeDetailSpider.newUrls) - 1
        if crawlUrls > 10000:
            crawlUrls = 10000

        for link in ResumeDetailSpider.newUrls[:crawlUrls]:
            urls = pd.DataFrame({"urls": [link]})
            insertCollection(db="thesisDb_Crawlers", col="resumeCrawler_resumeCrawledUrls", result=urls)

            yield Request(url=link, meta={"currentURL": link})

    def parse(self, response):
        # Position Name/ Resume Title
        title = response.css('h1::text')[1].get()

        # Location
        location = response.css('a::text')[1].get()

        # Posted On:
        postedOn = response.css('#PostedDate::text').get()

        # Resume
        resume = response.xpath('//div[@class="normalText"]')[0].extract()
        resumeText = BeautifulSoup(resume, "lxml").text

        profile = dict()
        profile['title'] = title
        profile['location'] = location
        profile['postedOn'] = postedOn
        profile['resume'] = resumeText
        timeOfCrawl = str(datetime.now())
        profile['timestamp'] = timeOfCrawl
        profile['profileLink'] = response.meta["currentURL"]

        # print(profile)
        urls = pd.DataFrame.from_dict([profile])
        insertCollection(db="thesisDb_Crawlers", col="resumeCrawler_profiles", result=urls)
        sleep(2)
