from pymongo import MongoClient
import pandas as pd


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


def insertCollection(db, col, result, drop=True):
    """Inserts the given data into the specified databases collection

    :param db: Database where the collection is
    :type db: string
    :param col: Collection where the data should be stored
    :type col: string
    :param result: Data that should be stored
    :type result: Pandas Dataframe
    """
    ## TODO Add a "db.drop(col)" before inserting, so everytime we insert a collection to the database, remove this particular collection

    result = result.to_dict("records")
    conn = MongoClient("localhost", 27017)
    connObj = conn[db][col]
    if drop:
        connObj.drop()
    connObj.insert_many(result)
