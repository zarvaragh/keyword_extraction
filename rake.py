import re
import pandas as pd
from rake_nltk import Rake

dt = pd.read_csv("Train.csv", nrows=5000)
dt["Text"] = (dt["Title"] + dt["Body"]).apply(lambda x: re.sub(r"(</?.*?>)|(\d|\W)+", " ", str(x).lower()))

rake = Rake()
dt["Keywords"] = ""
for i in range(len(dt)):
    rake.extract_keywords_from_text(dt["Text"].iloc[i])
    phrases = rake.get_ranked_phrases()
    dt.at[i, "Keywords"] = phrases[0] if phrases else ""