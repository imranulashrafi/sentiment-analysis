import os

import pymongo


class Database(object):
    username = os.environ.get("USERNAME")
    password = os.environ.get("PASSWORD")
    uri = "mongodb://{}:{}@ds147354.mlab.com:47354/webapp".format(username, password)
    database = None

    @staticmethod
    def initialize():
        client = pymongo.MongoClient(Database.uri)
        Database.database = client['webapp']

    @staticmethod
    def insert(collection, data):
        Database.database[collection].insert(data)

