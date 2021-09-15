import os
from werkzeug.utils import secure_filename
from JB_REC.connections.mongoConnection import getCollection, insertCollection, findRecord
from JB_REC.textprocessing.cleaning import Cleaner
import pandas as pd
from JB_REC.ner.annotation import Annotations
from flask import Flask, render_template, request

app = Flask(__name__, template_folder="JB_REC")


@app.route('/testingApp', methods=['GET'])
def testingApp():
    dataGlassdoor = getCollection("thesisDb_Crawlers", "glassdoorJobListings")

    insertCollection("thesisDb_Crawlers", "glassdoorListingFiltered", dataGlassdoor)
    print("Testing This APp")
    return "Testing App"


@app.route('/cleanJobListings', methods=['GET'])
def cleaningJobListingData():
    print("Request Received")

    dataGlassdoor = getCollection("thesisDb_Crawlers", "glassdoorJobListings")  # [x] Import Original Data from the
    # Crawler

    cleaner = Cleaner()  # Initialize Class

    dataGlassdoorFilter = Cleaner.replaceSpacesFilterColumns(dataGlassdoor)  # [x] Apply Space and Filter Changes
    insertCollection("thesisDb_Crawlers", "glassdoorListingFiltered", dataGlassdoorFilter)  # [x] Store the Data
    print("Inserted")
    dataGlassdoorAll = getCollection("thesisDb_Crawlers", "glassdoorListingFiltered")  # [x] Import Filtered Data
    colNames = {}
    for col in dataGlassdoorAll.columns:  # [x] Perform Cleaning Methods from the class
        dataGlassdoorAll[col] = dataGlassdoorAll[col].astype(str)
        dataGlassdoorAll[col] = dataGlassdoorAll[col].apply(cleaner.cleanHTMLTags)
        colNames[col] = str(col).replace(".", "_")

    dataGlassdoorAll = dataGlassdoorAll.rename(columns=colNames)

    insertCollection("thesisDb_Crawlers", "glassdoorListingsClean", dataGlassdoorAll)
    return "Data Cleaned"


@app.route('/form')
def form():
    return render_template('form.html')


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    # https://pythonbasics.org/flask-upload-file/
    #  https://www.youtube.com/watch?v=6WruncSoCdI&ab_channel=JulianNash
    print("Inside Upload")
    print(request)
    print(request.files['file'])
    if request.method == 'POST':
        if request.files:
            # File received from the form
            f = request.files['file']
            f.save(os.path.join("JB_REC/data/", secure_filename(f.filename)))
            df = pd.read_json("JB_REC/data/" + secure_filename(f.filename), lines=True)
            # Get Labels / Annotations from the File
            annotations = Annotations()
            set_of_skills = Annotations.getAllAnnotations(df)
            print(set_of_skills)

            sos = list()
            for skill in set_of_skills:
                print(findRecord("thesisDB_Annotation", "skillset", query={"skills": skill}))
                if not findRecord("thesisDB_Annotation", "skillset", query={"skills": skill}):
                    sos.append(skill)
            print(sos)
            if sos is not None:
                # Save Labels
                skillDf = pd.DataFrame(sos, columns=["skills"])
                insertCollection("thesisDB_Annotation", "skillset", skillDf, drop=False)

            # # Get All Labels from DB
            # skillT = getCollection("thesisDB_Annotation", "skillset")
            # skillT = skillT.drop_duplicates(subset=["skills"])
            # set_of_skills = skillT['skills'].tolist()
            #
            # dsDf = getCollection("thesisDb_Crawlers", "glassdoorListingsClean")
            # # Now annotate all the data
            # dsDf = Annotations.findAllSkills(dsDf, set(set_of_skills))
            # # Upload Data to MongoDB as Training for NER
            # insertCollection("thesisDB_Annotation", "AnnotatedJobListings", dsDf)

            return "File saved successfully"


@app.route('/annotateJobs', methods=['POST', 'GET'])
def annotateJobListings():
    print("Started Annotation")
    # Get All Labels from DB
    skillT = getCollection("thesisDB_Annotation", "skillset")
    skillT = skillT.drop_duplicates(subset=["skills"])
    set_of_skills = skillT['skills'].tolist()
    print("Data Extracted")

    dsDf = getCollection("thesisDb_Crawlers", "glassdoorListingsClean")
    # Now annotate all the data
    dsDf = Annotations.findAllSkills(dsDf, set(set_of_skills))
    print("Finished Annotation")
    # Upload Data to MongoDB as Training for NER
    insertCollection("thesisDB_Annotation", "AnnotatedJobListings", dsDf)
    print("Inserted Records")

    return "Records Annotated successfully"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
