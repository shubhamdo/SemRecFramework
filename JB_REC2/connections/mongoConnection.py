# [x] Connect to Mongo
# [x] Read the Glassdoor File
# [x] Load the Dataframe to MongoDB


from pymongo import MongoClient
import pandas as pd
import numpy as np


def getCollection(db, col):
    """Returns the data for a given database and collection

    Parameters
    ----------
    db : str
        Name of the database to be inserted to
    col : str
        Name of the collection to be inserted to

    Returns
    ----------
    pd.Dataframe
        Returns requested data from the collection

    """
    conn = MongoClient("localhost", 27017)
    collobj = conn[db][col]
    collection = pd.DataFrame(list(collobj.find({})))
    conn.close()
    return collection.copy()


def insertRow(db, col, result):
    """Inserts the given data into the specified databases collection

        Parameters
        ----------
        db : str
            Name of the database to be inserted to
        col : str
            Name of the collection to be inserted to
        result : Pandas Dataframe
            Data that is to be inserted
    """

    # result = result.to_dict("records")
    conn = MongoClient("localhost", 27017)
    connObj = conn[db][col]
    for x, row in result.iterrows():
        connObj.insert_one(row.to_dict())
    conn.close()


def insertCollection(db, col, result, drop=True):
    """Inserts the given data into the specified databases collection

        Parameters
        ----------
        db : str
            Name of the database to be inserted to
        col : str
            Name of the collection to be inserted to
        result : Pandas Dataframe
            Data that is to be inserted
        drop : bool, default=True
            Flag for dropping the given collection


    """

    # result = result.to_dict("records")
    conn = MongoClient("localhost", 27017)
    connObj = conn[db][col]
    if drop:
        connObj.drop()
    # connObj.insert_many(result)
    for x, row in result.iterrows():
        connObj.insert_one(row.to_dict())
    conn.close()


def getValue(db, col, query):
    conn = MongoClient("localhost", 27017)
    connObj = conn[db][col]
    result = connObj.find(query, {'_id': 0})
    # print(result)
    conn.close()
    return result


def findRecord(db, col, query):
    """Finds if the record exists in the database specified

        Parameters
        ----------
        db : str
            Name of the database to be inserted to
        col : str
            Name of the collection
        query : dict
            Filters and Query for the search

    """

    # result = result.to_dict("records")
    conn = MongoClient("localhost", 27017)
    connObj = conn[db][col]
    exist_count = connObj.find(query, {'_id': 0}).count()
    conn.close()
    if exist_count >= 1:
        return True
    else:
        return False


if __name__ == "__main__":
    pass
    # dataGlassdoor = pd.read_csv("H:/Thesis/Input_Data/glassdoor/glassdoor.csv",
    #                             na_filter=True)
    # insertCollection("thesisDb_Crawlers", "glassdoorJobListings", dataGlassdoor)
