from JB_REC2.connections.mongoConnection import getCollection, insertCollection
from langdetect import detect
import numpy as np
import re
import pandas as pd


class Cleaner:
    """
        Cleaner performs manipulation on the Text Data Provided for cleaning.
        This class deals with cleaning HTML Tags, spaces in the text that were acquired through crawling.

    """

    @staticmethod
    def cleanHTMLTags(text):
        """
        This function will be used to remove tags from the given input string

        Parameters
        ----------
        text : str
            Text to be cleaned for HTML Tags.

        Returns
        -------
        cleanText : str
            Cleaned Text after removal of HTML Tags.
        """
        if text is np.nan:
            text = ""
        htmlTags = re.compile('<.*?>')
        cleanText = re.sub(htmlTags, ' ', text)
        cleanText = re.sub("\n", ' ', cleanText)
        cleanText = re.sub("\r", ' ', cleanText)
        cleanText = re.sub(";", '.', cleanText)
        return cleanText

    @staticmethod
    def countMaxWordsInText(df):
        """
        This function used to count number of words from the given input string in the dataframe

        Parameters
        ----------
        df : Pandas Dataframe
            Contains 'job.description' series that is to be used as an input string

        Returns
        -------
        max_len : str
            Returns the number of words present for each string in the job.description in series
        """
        max_len = 0
        df = df['job.description'].astype(str)
        for jd in df['job.description']:
            if jd is np.nan:
                continue
            else:
                jdLen = len(jd.split())
                print(jdLen)
                if jdLen > max_len:
                    max_len = jdLen
        return max_len

    @staticmethod
    def replaceSpacesFilterColumns(df):
        """
        This function used to filter useful columns from all 164 original columns, filter null / na records
        from the dataset, replace spaces to apply the same filters

        Parameters
        ----------
        df : Pandas Dataframe
            Contains glassDoor Job Listings with 164 columns

        Returns
        -------
        df : Pandas Dataframe
            Returns the data after filtering and cleaning data
        """
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


        # Filter NA Records
        df = df[df["job.listingId.long"].notna()]
        df = df[df["job.description"].notna()]

        # Fix Type Issues
        df["job.description"] = df["job.description"].astype(str)
        df["job.listingId.long"] = df["job.listingId.long"].astype(np.int64)

        # Replace Spaces
        df = df.replace({r'\s+$': ' ', r'^\s+': ' '}, regex=True).replace(r'\n', ' ', regex=True)
        df = df[pd.notnull(df['job.description'])]

        return df


if __name__ == "__main__":

    # [x] Import Original Data from the Crawler
    dataGlassdoor = getCollection("thesisDb_Crawlers", "glassdoorJobListings")

    # Initialize Class
    cleaner = Cleaner()

    # [x] Apply Space and Filter Changes
    dataGlassdoor = Cleaner.replaceSpacesFilterColumns(dataGlassdoor)

    # [x] Store the Data
    insertCollection("thesisDb_Crawlers", "glassdoorListingFiltered", dataGlassdoor)

    # [x] Import Filtered Data
    dataGlassdoor = getCollection("thesisDb_Crawlers", "glassdoorListingsFiltered")

    # [x] Perform Cleaning Methods from the class
    for col in dataGlassdoor.columns:
        dataGlassdoor[col] = dataGlassdoor[col].astype(str)
        dataGlassdoor[col] = dataGlassdoor[col].apply(cleaner.cleanHTMLTags)

    insertCollection("thesisDb_Crawlers", "glassdoorListingsClean", dataGlassdoor)
