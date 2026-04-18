import os
from cerebras.cloud.sdk import Cerebras
import pandas as pd
import numpy as np
import json
from config_gen import generate_coalitional_conf, generate_divergent_conf, generate_minorty_conf, generate_uniform_conf
from strats import ADD, MPL, APP, LMS, MAJ, FAI
import random
import time
import re
from ollama import Client
from cerebras.cloud.sdk import RateLimitError
random.seed(time.time())


client_ol = Client()
client = Cerebras(
    api_key='YOUR_KEY'
)

configurations = {
    "coalitional": lambda: generate_coalitional_conf(n=5, m=10, r=5),
    "divergent":   lambda: generate_divergent_conf(n=5, m=10, r=5),
    "minority":    lambda: generate_minorty_conf(n=5, m=10, r=5),
    "uniform":     lambda: generate_uniform_conf(n=5, m=10, r=5),
}


strategies = {
    "ADD": lambda: ADD(rating_df),
    "MAJ": lambda: MAJ(result),
    "LMS": lambda: LMS(rating_df),
    "MPL": lambda: MPL(rating_df),
    "APP": lambda: APP(rating_df, threshold=3),
    "FAI": lambda: FAI(result),
}

explanations = {
    "ADD": "since it achieves the highest total rating among the available options",
    "MAJ": "since most group members like it among the available options",
    "LMS": "since no group members has a real problem with it among the available options",
    "MPL": "since it achieves the highest of all individual group members among the available options",
    "APP": "since it achieves the highest number of ratings which are above 3 among the available options",
    

}



system_message = {
        'role': 'system',
        'content': """
        You are a participant in a scientific study. 
        You will be presented with four scenarios related to four different groups of people. 
        For each of them, a software system will produce recommendations on the basis of the preferences of the group members. 
        Please read carefully the description of each scenario, and then answer the questions. 

        The statements are to be answered using a 7-point Likert scale:
        -3: strongly disagree
        -2: disagree
        -1: somehwat disagree
        0: neither agree, neither disagree
        1: somewhat agree
        2: agree
        3: strongly agree

        You are to return ONLY a valid JSON object, with no explanations or extra text. 
        Do not include markdown formatting (no ```json). 
        The JSON must have the following structure:
        {{
        "fairness": your reply,
        "consensus": your reply,
        "satisfaction": your reply
        }}
                """
                }





count = 0
goal = 1000
ers = []

iterations = 1 # if LLMs unstable, you might do multiple iterations

configuration_list = ['divergent', 'uniform', 'coalitional', 'minority']
expl_list = [0,1]
llm_name = "llama3.1-8b" # llama3.1-8b from Ollama or Cerebras
names= ['Alex', 'Bob', 'Carl', 'David', 'Elle']

errorcount = 0

file_path = 'results.csv'

if os.path.exists(file_path):
    df_results = pd.read_csv(file_path)
else:
    columns = ['iteration','configuration', 'strategy', 'expl', 'fairness', 'consensus', 'satisfaction', 'scenario', 'llm']
    df_results = pd.DataFrame(columns=columns)

while count <goal :
    try:    
        time.sleep(1)
        count +=1
        configuration = random.choice(configuration_list)

        matrix = configurations[configuration]()

        restaurants = matrix[0].keys()
        result = {r: [entry[r] for entry in matrix] for r in restaurants}
        
        person_four = names[3]
        rating_df = pd.DataFrame([
                        {"item": item, "rating": r}
                        for item, ratings in dict(result).items()
                        for r in ratings
                    ])
        for strategy in ['ADD', 'APP', 'LMS', 'MPL', 'MAJ', 'FAI']:
            if strategy =='FAI':
                result['names'] = names
            for expl in [0,1]:
                order = strategies[strategy]()
                already_visited = ", ".join(map(str, order[0:3]))
                recommendation = str(order[3])

                if expl == 1:
                    if strategy =='FAI':
                        explanation = f"since it is the user {person_four} turn and this is his favourite choice among the available options"
                    else:
                        explanation = explanations[strategy]
                else:
                    explanation = " "

                messages = []
                messages.append(system_message)

                scenario = {
        'role': 'user',
        'content': f"""
        Assume that there is a group of friends. Every month, a group decision is made by these friends to decide on a restaurant to have dinner together. 
        To select a restaurant for the dinner next month, the group again has to take the same decision. 
        In this decision, each group member explicitly rated ten possible restaurants using a 5-star rating scale (1: the worst, 5: the best). 
        The ratings given by group members are shown in the table below:

        ### BEGIN TABLE ###
        {result}
        ### END TABLE ###

        The group decided to avoid going in the same restaurant too often; hence, after a restaurant has been selected, it cannot be chosen again for the next 4 dinners. 
        **The last 3 restaurants visited are: {already_visited} **.
        Using the provided ratings, the system made a suggestion for the group on the basis of the preferences of the all the group members.
        **{recommendation}** has been recommended to the group {explanation}.

        Indicate your agreement to the following statements (7-point likert scale ranging from -3 to 3):

        ** 1.  Fairness **
        The group recommendation is fair to all group members.
        
        ** 2. Consensus **
        The group members will agree on the group recommendation.
        
        ** 3. Satisfaction **
        The group members will be satisfied with regard to the group recommendation.

        Reply only with the json object which includes your replies (integer 7-point likert scale -3 to 3). 
        """

    }


                messages.append(scenario)
                i=1
                while i<iterations+1:
                    chat_instance = messages
                    if llm_name=='gpt-oss:20b-cloud':
                        resp = client_ol.chat(llm_name, messages=messages)
                        out = resp['message']['content']
                    else:
                        resp = client.chat.completions.create(
                                    messages = chat_instance,
                                    model=llm_name,
                                    stream=False,
                                    max_completion_tokens=40000,
                                    temperature=0.5,
                                    top_p=0.8
                                )

                        out = resp.choices[0].message.content
                    if "<think>" in out or "</think>" in out:
                        out = re.sub(r'^.*?</think>', '', out, flags=re.DOTALL).strip()


                    out = json.loads(out)


                    df_results = pd.concat([df_results, pd.DataFrame([{
                        'iteration': i,
                        'configuration': configuration,
                        'strategy': strategy,
                        'expl': expl,
                        'fairness': out['fairness'],
                        'consensus': out['consensus'],
                        'satisfaction':out['satisfaction'],
                        'scenario': f"{result} | already visited: {already_visited} | recommendation: {recommendation} | explanation: {explanation}",
                        'llm': llm_name,
                    }])], ignore_index=True)

                    df_results.to_csv('results.csv', index=False)
                    i+=1
    except RateLimitError as e:
        errorcount +=1
        msg = str(e)
        ers.append(msg)
        if errorcount > 5:
            print('Time to stop. Error count above 5')
            print(ers)
            break
        
        if "high traffic" in msg.lower() or "queue_exceeded" in msg.lower():
            print(f"High traffic detected.")
            time.sleep(20)
            continue
