import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util, models
from flask import Flask, request
from flask_cors import CORS, cross_origin
# from sklearn.utils.tests.test_cython_blas import ORDER
# from JB_REC.graph.injectdata import skills
from JB_REC2.connections.neoconnection import connectToNeo4J
from py2neo import Node, Relationship
from JB_REC2.connections.mongoConnection import getCollection, getValue, insertCollection, insertRow
import ast
from JB_REC2.datageneration.annotation import Annotations

ann = Annotations()

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

graph = connectToNeo4J()


@app.route("/createuser", methods=['POST'])
@cross_origin()
def createUser():
    data = request.json
    print(type(data))
    # data = json.loads(data)
    print(data['email'])
    print(data['name'])
    print(data['password'])
    regNode = graph.nodes.match("user", email=data["email"]).all()
    response = {"flag": "1"}

    if len(regNode) > 0:
        response["flag"] = -1
        return response

    tx = graph.begin()
    node = Node("user", name=data["name"], email=data["email"], password=data["password"])
    tx.merge(node, primary_label="user", primary_key="email")
    # print("Transaction", tx)
    graph.commit(tx)

    return response


@app.route('/profileroles', methods=['POST', 'GET'])
@cross_origin()
def profileRoles():
    # Create a node for summary and relationship --> HAS_SUMMARY (with key --> email)
    # USER -- HAS_ROLE --> ROLE (properties)
    if request.method == 'GET':
        print("This is get method")
        # data = request.json
        data = dict()
        data['email'] = request.args.get('email')
        print(data)

        roleList = []
        roleNodes = graph.nodes.match("role", email=data["email"]).all()
        for eachRole in roleNodes:
            print("This is one role", eachRole)
            roleList.append(eachRole)

        response = dict()
        response['roles'] = roleList
        return response

    if request.method == 'POST':
        print("This is POST method")
        data = request.json
        print(data)
        title = data['title']
        email = data['email']
        company = data['company']
        location = data['location']
        startdate = data['startdate']
        enddate = data['enddate']
        sector = data['sector']
        description = data['description']

        # TODO IMP NER Operation on the description
        """
        - Pass the description to the NER Model
        - Get all the tags from the model 
        - Pass the description to pre-known keywords
        - Get all the tags
        - Find all distinct tags
        - Store the tags in One Node for showing in profile mode, to edit it
        """

        roleNodes = graph.nodes.match("role", email=email).all()

        corpus = description
        # Find all skills in the descriptions
        for eachRole in roleNodes:
            title = eachRole['title']
            lDescription = eachRole['description']
            corpus = corpus + " " + title + " " + lDescription + " "

        # Create a query for getting recommendations for the common keywords / skills
        # Get All Labels from DB
        userNode = graph.nodes.match("user", email=email).first()

        ann.annotateCorpus(corpus, userNode)

        # Create Role Node
        tx = graph.begin()
        node = Node("role", title=title, email=email, company=company, location=location, startdate=startdate,
                    enddate=enddate,
                    sector=sector, description=description)
        tx.create(node)
        graph.commit(tx)

        # Create Relationship
        tx = graph.begin()
        skNode = graph.nodes.match("user", email=email).first()
        ab = Relationship(skNode, "HAS_ROLE", node)
        tx.merge(ab)
        graph.commit(tx)

        return "Success"


@app.route('/crecdetails', methods=['POST'])
@cross_origin()
def crecDetails():
    print("Cold Recommendation Details Called: crecDetails")
    data = request.json
    print(type(data))

    jobList = []
    for job in data:
        print(job)
        jobId = job['jobId']
        jobNode = graph.nodes.match("JD", jobId=jobId).first()
        # print(dict(jobNode))
        # print(str(jobNode['jobId']))
        values = getValue("thesisDb_Crawlers", "glassdoorListingsClean", {'job_listingId_long': str(jobNode['jobId'])})
        # print(list(values))
        for x in list(values):
            jobList.append(x)

    # jobList = map(dict, set(tuple(x.items()) for x in jobList))
    # print(jobList)
    jobList = [ast.literal_eval(el1) for el1 in set([str(el2) for el2 in jobList])]
    print(jobList)
    response = dict()
    response['data'] = jobList
    return response


@app.route('/applyjob', methods=['POST'])
@cross_origin()
def applyJob():
    print("Cold Recommendation Details Called: crecDetails")
    data = request.json
    jobId = data['job']['job_listingId_long']
    print(data)
    userId = data['userId']
    print(type(jobId), int(jobId), userId)

    userNode = graph.nodes.match("user", email=userId).first()
    jobNode = graph.nodes.match("JD", jobId=int(jobId)).first()

    print(jobNode, userNode)

    tx = graph.begin()
    ab = Relationship(userNode, "APPLIED", jobNode)
    tx.merge(ab)
    graph.commit(tx)

    return "Success"


@app.route('/crec', methods=['GET'])
def coldRecommendations():
    # Get the email address
    data = dict()
    data['email'] = request.args.get('email')
    # print(data)
    email = data['email']
    print(email)

    # Return the list of list in 10-15 records
    p1 = "MATCH (u:user {email:'"
    p2 = r"{}'".format(email)
    p3 = "})<-[:SKILLOF]-(skills)-[:SKILLOF]->(otherGroup:JD)    RETURN " \
         "otherGroup.jobId as jobId, " \
         "COUNT(skills) AS topicsInCommon " \
         "ORDER BY topicsInCommon DESC    " \
         "LIMIT 1000 "
    # "COLLECT(skills.skill) As topics    " \
    # "otherGroup.`header.jobTitle` as jobTitle, " \
    # "otherGroup.`job.description` as jobDescription," \
    print(p1 + p2 + p3)
    recommendations = graph.run(p1 + p2 + p3)
    print(type(recommendations))

    response = dict()
    recList = []
    for rec in recommendations:
        # print(rec.keys())
        # ['otherGroup.`header.jobTitle`', 'otherGroup.jobId', 'topicsInCommon', 'topics']
        recList.append(dict(rec))

    lol = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]
    recLol = lol(recList, 10)
    print(recLol)

    response['data'] = recLol

    return response


@app.route('/srec', methods=['POST'])
def similarRecommendations():
    # Get the email address
    data = dict()
    data = request.json
    # print(data)
    jobId = data['jobId']
    print(jobId)

    # Return the list of list in 10-15 records
    p1 = "MATCH (j:JD {jobId: "
    p2 = r"{}".format(jobId)
    p3 = "})<-[:SKILLOF]-(skills)-[:SKILLOF]->(otherGroup:JD)    RETURN " \
         "otherGroup.jobId as jobId, " \
         "COUNT(skills) AS topicsInCommon " \
         "ORDER BY topicsInCommon DESC    " \
         "LIMIT 10 "
    print(p1 + p2 + p3)
    recommendations = graph.run(p1 + p2 + p3)

    response = dict()
    recList = []
    for rec in recommendations:
        # print(rec.keys())
        # ['otherGroup.`header.jobTitle`', 'otherGroup.jobId', 'topicsInCommon', 'topics']
        rec = dict(rec)
        jobId = rec['jobId']
        values = getValue("thesisDb_Crawlers", "glassdoorListingsClean", {'job_listingId_long': str(jobId)})
        jobList = []
        for x in list(values):
            jobList.append(x)

        jobList = [ast.literal_eval(el1) for el1 in set([str(el2) for el2 in jobList])]
        print(jobList)
        for x in jobList:
            recList.append(x)
            print("This is a job", x)

    response['data'] = recList
    print(response)

    return response


@app.route('/jobdetails', methods=['GET'])
def jobDetails():
    # Get the Job Id
    # Find the job with same details
    # Create a query for getting recommendations for the common keywords / skills with this job and another jobs
    # Return the job and the similar jobs
    jobId = request.args.get("jobId")
    print("Job Details", jobId)

    values = getValue("thesisDb_Crawlers", "glassdoorListingsClean", {'job_listingId_long': jobId})
    jobList = []
    for x in list(values):
        jobList.append(x)
        print("JD", x)

    jobList = [ast.literal_eval(el1) for el1 in set([str(el2) for el2 in jobList])]
    print("JD", jobList)
    response = dict()
    response['data'] = jobList

    return response


@app.route('/collab', methods=['POST'])
def collaborativeFilter():
    # Get the email address
    # Query to find user who applied same jobs
    # Find the jobs that you did not apply
    # Set a limit of 500

    pass


@app.route('/profilesummary', methods=['POST', 'GET'])
@cross_origin()
def profileSummary():
    if request.method == 'GET':
        print("This is get method")
        # data = request.json
        data = dict()
        data['email'] = request.args.get('email')
        print(data)

        roleNodes = graph.nodes.match("summary", email=data["email"]).first()

        response = dict()
        response['roles'] = [roleNodes]
        return response

    if request.method == 'POST':
        print("This is POST method")
        data = request.json
        print(data)
        role = data['title']
        email = data['email']
        summary = data['summary']
        avatar = "/avatar-1.png"
        name = data['name']

        summaryNode = graph.nodes.match("summary", email=data["email"]).first()
        if summaryNode:
            # Create Role Node
            tx = graph.begin()
            node = Node("summary", role=role, email=email, summary=summary, name=name, avatar=avatar)
            tx.merge(node, primary_label="summary", primary_key="email")
            graph.commit(tx)

        else:
            # Create Role Node
            tx = graph.begin()
            node = Node("summary", role=role, email=email, summary=summary, name=name, avatar=avatar)
            tx.create(node)
            graph.commit(tx)

            # Create Relationship
            tx = graph.begin()
            skNode = graph.nodes.match("user", email=email).first()
            ab = Relationship(skNode, "HAS_SUMMARY", node)
            tx.create(ab)
            graph.commit(tx)

        return "Success"


@app.route("/login", methods=['POST'])
@cross_origin()
def loginUser():
    data = request.json
    print(type(data))
    # data = json.loads(data)
    print(data['email'])
    print(data['password'])

    response = {"flag": 1}
    loginNode = graph.nodes.match("user", email=data["email"]).all()
    password = loginNode[0]["password"]
    if password == data["password"]:
        pass
    else:
        response["flag"] = 0

    # tx = graph.begin()
    # node = Node("user", name=data["name"], email=data["email"], password=data["password"])
    # tx.merge(node, primary_label="user", primary_key="email")
    # graph.commit(tx)
    # print(graph.commit())

    return response


@app.route('/addjob', methods=['POST'])
@cross_origin()
def addJob():
    if request.method == 'POST':
        data = request.json
        print(data)
        title = data['jobTitle']
        employerName = data['employerName']
        location = data['location']
        sector = data['sector']
        jobDescription = data['jobDescription']
        industry = data['industry']
        companySize = "nan"

        highestId = graph.nodes.match("ID").first()

        if highestId is None:
            tx = graph.begin()
            node = Node("ID", hI=1, user="admin")
            tx.create(node)
            graph.commit(tx)

        highestId = graph.nodes.match("ID").first()
        highestId = dict(highestId)
        id = highestId['hI']
        id = id + 1

        tx = graph.begin()
        idNode = Node("ID", hI=id, user="admin")
        tx.merge(idNode, primary_key="user", primary_label="ID")
        graph.commit(tx)

        tx = graph.begin()
        jobNode = Node("JD",
                       **{"jobId": id, "header.jobTitle": title, "job.description": jobDescription, "clusters": -12})
        tx.create(jobNode)
        graph.commit(tx)

        gJobNode = graph.nodes.match("JD", jobId=id).first()

        # Call the annotation methods
        ann.annotateCorpus(title + " " + jobDescription, gJobNode, graph)
        # Load the rest of the data to mongodb to access it

        record = {"job_listingId_long": str(id),
                  "header_jobTitle": title,
                  "header_location": location,
                  "job_description": jobDescription,
                  "job_jobSource": "Company Employee",
                  "map_country": location,
                  "map_employerName": employerName,
                  "map_location": location,
                  "overview_industry": industry,
                  "overview_sector": sector,
                  "overview_size": companySize,
                  "overview_description": "nan"}

        record = pd.DataFrame([record])
        insertRow("thesisDb_Crawlers", "glassdoorListingsClean", record)

        print(title, employerName, location, sector, jobDescription, industry, id)

        response = dict()
        response['id'] = id
        return response


@app.route('/appliedjobs', methods=['GET'])
def appliedJobs():
    # Get the email address
    data = dict()
    data['email'] = request.args.get('email')
    email = data['email']

    # Return the list of list in 10-15 records
    p1 = "MATCH (u:user {email:'"
    p2 = r"{}'".format(email)
    p3 = "})-[:APPLIED]-(j:JD) RETURN j"

    jobs = graph.run(p1 + p2 + p3)
    jobList = []
    for x in jobs:
        x = dict(x['j'])
        jobId = x['jobId']

        values = getValue("thesisDb_Crawlers", "glassdoorListingsClean", {'job_listingId_long': str(jobId)})
        for y in list(values):
            print(y)
            jobList.append(y)

    jobList = [ast.literal_eval(el1) for el1 in set([str(el2) for el2 in jobList])]
    print("Final", jobList)

    response = dict()
    response['data'] = jobList
    return response


@app.route('/stscompare', methods=['POST'])
def upload():
    data = request.json
    print(data)
    a = str(data['sentence1'])
    b = str(data['sentence2'])

    print(a, b)
    # [x] Import Model
    modelpath = 'H:/Thesis/Output_Data/TrainingData/25/Roberta'
    word_embedding_model = models.Transformer(model_name_or_path=modelpath
                                              , max_seq_length=512)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model.cuda()
    # [x] Pass the Records
    # sentences1 = [positiveRecords['inputA'].iloc[0]]
    # sentences2 = [positiveRecords['inputB'].iloc[0]]
    sentences1 = [a]
    sentences2 = [b]

    # [x] Write Inference Code
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    # [x] Get Cosin value
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    ag = []
    # Output the pairs with their score
    for i in range(len(sentences1)):
        # print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
        print("{} \t\n {} \t\n Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
        ag.append(cosine_scores[i][i])
    return str(ag)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
