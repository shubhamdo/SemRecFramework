import json
import re
import pandas as pd
from ast import literal_eval
from JB_REC2.connections.mongoConnection import insertCollection, getCollection
from py2neo import Relationship


class Annotations:

    @staticmethod
    def getAllAnnotations(df):
        """
        After the annotations from some small sample records, we need to use those to create a set of labels,
        so we can use it to label / annotate other records

        Parameters
        __________
        df:
            Dataframe containing the annotated labels and the records
        Returns
        __________
        list:
            set of skills extracted from the list

        """
        df['label'] = df['label'].astype(str)
        set_of_skills = set()
        for col, item in df.iterrows():
            id = item[0]
            data = item[1]
            labels = item[2]
            labels = literal_eval(labels)
            # print(f"Labels variable type: {type(labels)}")
            # print(id)
            # print(data)
            # print(labels)

            for eachLabel in labels:
                # print(eachLabel)
                skill = data[eachLabel[0]:eachLabel[1]].strip()
                # removeSpecialChar = r"[^0-9a-zA-Z]"
                skill = skill.replace("++", "\+\+")  # multiple repeat at position 2 issue
                set_of_skills.add(skill.lower())
        return set_of_skills


    @staticmethod
    def findAllSkills(dsDf, soSkills):
        """
        For each of the job description get all the annotations for the set of annotations acquired from annotations

        Parameters
        __________
        jobDes:
            string of the job description
        soSkills:
            all annotated set of skills

        Returns
        __________
        list:
            list of list containing start, end indexes and tag of the annotation
        """

        joblabels = []
        for index, row in dsDf.iterrows():
            print(index)
            jobDes = row[2].lower()
            # print(jobDes)
            label = []
            for skillLabel in soSkills:
                # print('OOOUT Match + ', skillLabel)
                for match in re.finditer(pattern=" " + skillLabel + " ", string=jobDes):
                    label.append([match.start() + 1, match.end() - 1, "SKILL"])
            if len(label) == 0:
                joblabels.append("")
            else:
                joblabels.append(label)

        dsDf['job.description.labels'] = pd.Series(data=joblabels)
        dsDf = dsDf[dsDf['job.description.labels'] != ""]
        return dsDf

    @staticmethod
    def annotateCorpus(jString, jNode, graph):

        skillT = getCollection("thesisDB_Annotation", "skillset")
        skillT = skillT.drop_duplicates(subset=["skills"])
        set_of_skills = skillT['skills'].tolist()

        # TODO NER Models - Fetch Skills which are in the description

        jString = jString.replace("\n", " ").lower()
        for skill in set_of_skills:
            ptrnSkill = r"\b{}\b".format(skill)
            for match in re.finditer(pattern=ptrnSkill, string=jString):
                tx = graph.begin()
                print(skill)
                skNode = graph.nodes.match("skills", skill=skill).first()
                if skNode is None:
                    continue
                ab = Relationship(skNode, "SKILLOF", jNode)
                tx.merge(ab)
                graph.commit(tx)

    @staticmethod
    def JSONGenerator(df, filename):
        """
        Create JSONL file for NER training using the job descriptions

        Parameters
        __________
        df:
            Dataframe with the job descriptions
        filename
            filepath + filename for the file to be created
        """
        with open(filename, "w", encoding='utf-8') as tf:
            df = df[['job.listingId.long', 'job.description', 'job.description.labels']]
            for index, row in df.iterrows():
                jdid = row[0]
                jd = row[1]
                jdl = row[2]
                jsString = {"id": jdid, "data": jd, "label": jdl}
                # tf.write(str(jsString) + '\n')
                json.dump(jsString, tf)
                tf.write('\n')


if __name__ == "__main__":
    # Annotated Data
    df = pd.read_json("H:/Thesis/Output_Data/annotation_jsons/softwareeng/admin.jsonl", lines=True)

    # Data Scientist
    # df = pd.read_json("H:/Thesis/Output_Data/annotation_jsons/datascientist.jsonl", lines=True)

    df['label'] = df['label'].astype(str)
    annotations = Annotations()
    set_of_skills = Annotations.getAllAnnotations(df)

    # Job Description Data Import Cluster Number - 5 Software Engineer
    # dsDf = pd.read_csv("H:/Thesis/Output_Data/cluster_all_5.csv", delimiter=";")
    # dsDf = dsDf[dsDf.notna()]

    dsDf = getCollection("thesisDb_Crawlers", "glassdoorListingsClean")
    dsDf = dsDf[dsDf.notna()]

    # Job Description Data Import Cluster Number - 4 Data Scientist Records
    # dsDf = pd.read_csv("H:/Thesis/Output_Data/cluster_all_4.csv", delimiter=";")
    # dsDf = dsDf[dsDf.notna()]

    # Remove Duplicates
    dsDf.drop_duplicates(inplace=True)
    dsDf = dsDf[dsDf['job.description'].notna()]

    # Get New Annotations
    # dsDf, kwDf = Annotations.findAllSkills(dsDf, set_of_skills)
    dsDf = Annotations.findAllSkills(dsDf, set_of_skills)

    # Create Annotation JSON File
    Annotations.JSONGenerator(dsDf, "H:/Thesis/Output_Data/se.jsonl")

    # findAllSkills(dsDf['job.description'][0], set_of_skills)
    # TODO Get the annotation for 5 more clusters
    # Load the data to neo4J after getting the words
