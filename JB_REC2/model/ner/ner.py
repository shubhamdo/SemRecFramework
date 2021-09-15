import pandas as pd
from ast import literal_eval


def ConflictingEnts(TRAIN_DATA):
    for eachRecord in TRAIN_DATA:
        ent = []
        for eachLabel in eachRecord[1]['entities']:
            start = eachLabel[0]
            end = eachLabel[1]
            label = eachLabel[2]
            len = end - start
            ent.append((start, end, label, len))

        df = pd.DataFrame(ent, columns=['start', 'end', 'tag', 'len'])
        startDf = pd.DataFrame(columns=['start', 'end', 'tag', 'len'])
        rejectDf = pd.DataFrame(columns=['start', 'end', 'tag', 'len'])

        for i, x in df.iterrows():
            start = x[0]
            end = x[1]
            if ((rejectDf['start'] == start) & (rejectDf['end'] == end)).any():
                continue
            tagDf = df[df['start'] == start]
            if tagDf.shape[0] > 1:
                tagDf = tagDf.sort_values(by='len', ascending=False)
                removeRows = tagDf.iloc[1:, ]
                tagDf = tagDf.iloc[:1, ]
                startDf = startDf.append(tagDf, ignore_index=False)
                # rejectDf = rejectDf.append(removeRows, ignore_index=False)
            else:
                startDf = startDf.append(tagDf, ignore_index=False)
        startDf = startDf.drop_duplicates()

        endDf = pd.DataFrame(columns=['start', 'end', 'tag', 'len'])
        for i, x in startDf.iterrows():
            end = x[1]
            tagDf = startDf[startDf['end'] == end]
            if tagDf.shape[0] > 1:
                tagDf = tagDf.sort_values(by='len', ascending=False)
                tagDf = tagDf.iloc[:1, ]
                endDf = endDf.append(tagDf, ignore_index=True)
            else:
                endDf = endDf.append(tagDf, ignore_index=True)
        endDf = endDf.drop_duplicates()

        centerDf = pd.DataFrame(columns=['start', 'end', 'tag', 'len'])
        for i, x in endDf.iterrows():
            end = x[1]
            start = x[0]
            temp = endDf[endDf['start'] < end]
            temp = temp[end < temp['end']]
            temp2 = endDf[endDf['start'] < start]
            temp2 = temp2[start < temp2['end']]
            frames = [temp, temp2]
            tagDf = pd.concat(frames)

            if tagDf.shape[0] > 1:
                tagDf = tagDf.sort_values(by='len', ascending=False)
                tagDf = tagDf.iloc[:1, ]
                centerDf = centerDf.append(tagDf, ignore_index=True)
            else:
                centerDf = centerDf.append(tagDf, ignore_index=True)
        centerDf = centerDf.drop_duplicates()
        centerDf = centerDf.drop(['len'], axis=1)
        # print(endDf)

        eachRecord[1]['entities'] = list(centerDf.itertuples(index=False, name=None))

    return TRAIN_DATA


df = pd.read_json("H:/Thesis/Output_Data/ner/datasci.json", lines=True)
se = pd.read_json("H:/Thesis/Output_Data/ner/se.jsonl", lines=True)
# Concatenate Data
frames = [df, se]
df = pd.concat(frames)
del se

# [] Create Correct Data Format
TRAIN_DATA = []
for item, row in df.iterrows():
    labels = literal_eval(str(row[2]))
    entities = []
    for label in labels:
        entities.append(tuple(label))
    TRAIN_DATA.append((row[1], {'entities': entities}))

# TRAIN_DATA = TRAIN_DATA[10:12]
TRAIN_DATA = ConflictingEnts(TRAIN_DATA)

from tqdm import tqdm
import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en")  # load a new spacy model
db = DocBin()  # create a DocBin object
count = 0
for text, annot in tqdm(TRAIN_DATA):  # data in previous format
    # print(text)
    doc = nlp.make_doc(text)  # create doc object from text
    ents = []
    for start, end, label in annot["entities"]:  # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        # print(span)
        if span is None:
            print("Skipping entity")
        elif span in ents:
            pass
        else:
            ents.append(span)
        # print(ents)
    try:
        doc.ents = ents  # label the text with the ents
        db.add(doc)
    except Exception as e:
        count += 1
        print(e)

print(count)

db.to_disk("H:/Thesis/Output_Data/ner/model/train.spacy")
print(spacy.__version__)
# s = [(
#     'GroupM is the world&rsquo.s largest media investment company and are a part of WPP. In fact, we are responsible for one in every three ads you see globally. We are currently looking for a Account Manager. The role will be responsible to drive the development of digital marketing research as a form of service or product for clients as well as a tool to support the growth of digital business for GroupM agencies    Reporting of the role    This role reports to Mindshare Lead     3 best things about the job:    Our ideal team member will have the mathematical and statistical expertise you&rsquo.d expect.   You will join a team of data specialists, but will &ldquo.slice and dice&rdquo. data using your own methods, creating new visions for the future.   Insightful data to power our systems and solutions     In this role, your goals will be:  In three months:    Communicate analytic solutions to stakeholders and implement improvements as needed to operational systems   Implement analytical models into production by collaborating with software developers and machine learning engineers.     In six months:    Devise and utilize algorithms and models to mine big data stores, perform data and error analysis to improve models, and clean and validate data for uniformity and accuracy   Analyse data for trends and patterns, and Interpret data with a clear objective in mind     In 12 months:    Work as the lead data strategist, identifying and integrating new datasets that can be leveraged through our product capabilities and work closely with the digital specialist team to strategize and execute the development of data products   Execute analytical experiments methodically to help solve various problems and make a true impact across various domains and industries   Identify relevant data sources and sets to mine for client business needs, and collect large structured and unstructured datasets and variables     What your day job looks like at Mindshare:    Collaborate with connection planning and digital specialists to develop an understanding of needs   Research and devise innovative statistical and machine learning models for data analysis     What you&rsquo.ll bring:    Communicate findings to all stakeholders, using relevant data visualisations   Enable smarter business processes&mdash.and implement analytics for meaningful insights   Keep current with technical and industry developments     Minimum qualifications:    Bachelor&rsquo.s degree in statistics, applied mathematics, or related discipline   5+ years&rsquo. experience in data science   Proficiency with data mining, mathematics, and statistical analysis   Advanced pattern recognition, recommendation engines and predictive modelling experience   Experience with Power BI, Tableau, SQL, and programming languages (i.e., Java/Python/R, SAS)   Comfort working in a dynamic, research-oriented group with several ongoing concurrent projects    More about Mindshare    We were born in Asia in 1997, a start up with a desire to change the media world. Now we are a global agency with more than 7,000 employees in 116 offices across 86 countries, operating as one team - #teammindshare. We believe that in today&rsquo.s world, everything begins and ends in media. We aim to be our clients&rsquo. lead business partner, to grow their business, and drive profitability through adaptive and inventive marketing. We do this through speed, teamwork and provocation and by operating as a network of networks rather than a rigid hierarchy. We create new things and have fun doing it. Whenever and wherever you join us, you open a door to opportunities in any and of all aspects of media, technology and innovation. We will support you, recognize you and reward you, making Mindshare the place where you do the best work of your career.    Mindshare APAC has won 500 awards in the last year alone, including &ldquo.Agency Network of the Year 2017&rdquo. by both the MMA SMARTIES&trade. and Campaign Asia for the fifth and third consecutive year, respectively. Mindshare is part of GroupM, the media investment management arm of WPP, the world&rsquo.s leading communications services group. To learn more about Mindshare and our philosophy of Original Thinking, visit us at www.mindshareworld.com and follow us on Twitter @mindshare and facebook.com/mindshareapac and linkedin.com/company/mindshare.    About Indonesia    Indonesia is one of the fastest growing and most dynamic markets in the world, with a population of more than 260 million. It&rsquo.s stable political climate and increasing consumer disposable income has meant that Indonesia is now firmly on the radar for MNCs seeking to invest in emerging markets.    Whilst the Indonesian media landscape is still dominated by terrestrial TV, with 65% of all advertising spend, digital is now poised to go through a period of exponential growth. There are more than 70 million internet users in Indonesia and 70% of them are accessing the web via mobile. As a marketer, the opportunity to transform a media landscape on such a scale is unprecedented in the APAC region.',
#     {'entities': [(2328, 2337, 'SKILL'), (247, 258, 'SKILL'), (1611, 1622, 'SKILL'), (566, 577, 'SKILL'),
#                   (2105, 2116, 'SKILL'), (2625, 2636, 'SKILL'), (1067, 1083, 'SKILL'), (2121, 2137, 'SKILL'),
#                   (2496, 2508, 'SKILL'), (2608, 2620, 'SKILL'), (987, 997, 'SKILL'), (1650, 1660, 'SKILL'),
#                   (1136, 1146, 'SKILL'), (1166, 1174, 'SKILL'), (1206, 1214, 'SKILL'), (2154, 2162, 'SKILL'),
#                   (2637, 2645, 'SKILL'), (262, 269, 'SKILL'), (376, 383, 'SKILL'), (1557, 1564, 'SKILL'),
#                   (2015, 2022, 'SKILL'), (4813, 4820, 'SKILL'), (2625, 2645, 'SKILL'), (451, 458, 'SKILL'),
#                   (2563, 2575, 'SKILL'), (2149, 2162, 'SKILL'), (2705, 2725, 'SKILL'), (4087, 4097, 'SKILL'),
#                   (416, 425, 'SKILL')]})]
