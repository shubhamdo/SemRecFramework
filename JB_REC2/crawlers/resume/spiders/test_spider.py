# import scrapy
# from bs4 import BeautifulSoup
# import pandas as pd
# # from resume.resume.mongoConnection import insertCollection
#
# keywords = pd.read_csv("H:/Thesis/Input_Data/keywords.csv")
# locations = pd.read_csv("H:/Thesis/Input_Data/locations.csv")
#
# for index, row in keywords.iterrows():
#     for index, rowLoc in locations.iterrows():
#         print(row[0], rowLoc[0])
#
# class ResumeSpider(scrapy.Spider):
#     name = "resume"
#     start_urls = [
#         'https://www.postjobfree.com/resume/adk9m3/data-analyst-sales-morristown-nj'
#     ]
#
#     def parse(self, response):
#         # Position Name/ Resume Title
#         response.css('h1::text')[1].get()
#
#         # Location
#         response.css('a::text')[1].get()
#
#         # Posted On:
#         response.css('#PostedDate::text').get()
#
#         # Resume
#         resume = response.xpath('//div[@class="normalText"]')[0].extract()
#         resumeText = BeautifulSoup(resume, "lxml").text
#         print(resumeText)
