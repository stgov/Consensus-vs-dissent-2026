import json
import ast
import re
import random
import time
import pathlib 
from cerebras.cloud.sdk import Cerebras
from sys import exit

from ollama import Client

client_ol = Client()
client = Cerebras(
    api_key='YOUR_KEY'
)
random.seed(time.time())
llm_name = "deepseek-v3.1:671b-cloud"


def generate_reasoning_prompt(restaurant_name, ratings, visited_restaurants, human_scores):
    """
    Generates the prompt 
    """
    prompt = f"""
You are a reasoning generator for group restaurant recommendations. 

Scenario:
- A group of friends is choosing a restaurant for dinner.
- Each member rated 10 possible restaurants on a 1-5 star scale. Meaning 3 is average, above 3 good, below 3 poor.
- The last 3 restaurants visited are: {', '.join(visited_restaurants)}.
- The group has decided not to visit restaurants they have already visited. These are no longer an option.
- Given the previously visited restaurants, the system suggests {restaurant_name} as the option the group will go to. Humans are NOT told how this suggestion was created — they only see the outcome and the ratings.
- Ratings for this restaurant: {ratings}.

Human-labeled scores (these are gold and must not be changed):
- Fairness: {human_scores['fairness']}
- Consensus: {human_scores['consensus']} 
- Satisfaction: {human_scores['satisfaction']}

Scores meaning:
-3: strongly disagree
-2: disagree
-1: somewhat disagree
0: neither agree nor disagree
1: somewhat agree
2: agree
3: strongly agree

Task:
Generate reasoning for each score (fairness, consensus, satisfaction) following these rules:
1. The reasoning must fully justify the gold human-labeled score.
2. Explicitly reference specific members and their ratings when relevant (e.g., "member 1 rated it 5 stars").
3. Mention briefly what could improve the score.
4. Use natural, human-like phrasing. Vary sentence structures and explain using casual language, just like a random participant in a user study.
5. Keep reasoning concise (2-3 sentences per dimension).
6. Start each reasoning with "Score [X] [your reasoning]" and align tone with the numeric score (-3 very negative, 3 very positive).
7. Avoid repeating or altering the human gold scores.

Output format (**do not forget to add the human gold label scores!**):

Fairness reasoning: ...
Consensus reasoning: ...
Satisfaction reasoning: ...
Scores: 
Fairness: {human_scores['fairness']}
Consensus: {human_scores['consensus']} 
Satisfaction: {human_scores['satisfaction']}
"""


    return prompt


def update_dataset_streaming(input_file, output_file):
    """
   main function for generating human-like reasoning grounded in human ground truth.
    """
    output_path = pathlib.Path(output_file)
    start_line = 0

    if output_path.exists():
        with output_path.open('r') as f:
            for count, _ in enumerate(f):
                start_line = count + 1
        
        if start_line > 0:
            print(f" RESUMING PROCESSING ")
            print(f"{start_line} items already processed in '{output_file}'. Starting from line {start_line + 1} of the input file.")
        else:
            print(f"Starting fresh. Output file '{output_file}' is empty.")
    else:
        print(f"Starting fresh. Output file '{output_file}' does not exist yet.")


    with open(input_file, 'r') as f_in, open(output_file, 'a') as f_out:
        
        for i, line in enumerate(f_in):
            
            if i < start_line:
                continue

            try:
                item = json.loads(line)
                
                user_msg = next(m for m in item['messages'] if m['role'] == 'user')
                assistant_msg = next(m for m in item['messages'] if m['role'] == 'assistant')

                content = user_msg['content']
                table_start = content.find("### BEGIN TABLE ###") + len("### BEGIN TABLE ###")
                table_end = content.find("### END TABLE ###")
                table_str = content[table_start:table_end].strip()
                ratings_table = ast.literal_eval(table_str)

                rec_match = re.search(r"\*\*(rest_\d+)\*\*", content)
                if rec_match:
                    recommended_restaurant = rec_match.group(1)
                else:
                    raise ValueError("Could not find recommended restaurant in user message.")

                visited_match = re.search(r"The last 3 restaurants visited are:\s*(.*?)(?:\.|\n)", content)
                last_visited = [r.strip() for r in visited_match.group(1).split(",")] if visited_match else []

                fairness_match = re.search(r'Fairness:\s*([-]?\d)', assistant_msg['content'])
                consensus_match = re.search(r'Consensus:\s*([-]?\d)', assistant_msg['content'])
                satisfaction_match = re.search(r'Satisfaction:\s*([-]?\d)', assistant_msg['content'])
                
                if not all([fairness_match, consensus_match, satisfaction_match]):
                     raise ValueError(f"Could not parse all scores for item on line {i}")

                fairness = int(fairness_match.group(1))
                consensus = int(consensus_match.group(1))
                satisfaction = int(satisfaction_match.group(1))
                human_scores = {"fairness": fairness, "consensus": consensus, "satisfaction": satisfaction}

                prompt_text = generate_reasoning_prompt(
                    restaurant_name=recommended_restaurant,
                    ratings=ratings_table[recommended_restaurant],
                    visited_restaurants=last_visited,
                    human_scores=human_scores
                )
                max_retries = 5
                initial_delay = 1
                resp = None

                for attempt in range(max_retries):
                    print(llm_name)
                    try:
                        if llm_name=='deepseek-v3.1:671b-cloud':
                            resp2 = client_ol.chat(llm_name, messages=[{"role": "user", "content": prompt_text}], options={"temperature":0.9, 'top_p':0.95})
                            out = resp2['message']['content']
                            break
                        else:
                            resp = client.chat.completions.create(
                                messages=[{"role": "user", "content": prompt_text}],
                                model=llm_name,
                                stream=False,
                                max_completion_tokens=750,
                                temperature=0.95,
                                top_p=0.95,
                                #top_k=100
                            )
                            break 
                    except Exception as e:
                            if attempt < max_retries - 1:
                                delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)
                                print(f"API Error (Attempt {attempt+1}/{max_retries}):{e} Retrying in {delay:.2f} seconds...")
                                time.sleep(delay)
                            else:
                                exit()
                                print(f"Failed to get response after {max_retries} attempts for {recommended_restaurant}. Error: {e}")
                                resp = None 


                if resp:
                    reasoning_text = resp.choices[0].message.content
                    assistant_msg['content'] = reasoning_text
    
      
                    f_out.write(json.dumps(item) + "\n")
                    f_out.flush()  
                    print(f"Successfully processed and updated item {i+1} for {recommended_restaurant}")
                elif out:
                    assistant_msg['content'] = out
                    f_out.write(json.dumps(item) + "\n")
                    f_out.flush() 
                    print(f"Successfully processed and updated item {i+1} for {recommended_restaurant}")
                else:   
                    print(f"Skipping line {i+1} ({recommended_restaurant}) due to API failure.")


            except Exception as e:
                print(f"Error processing item on line {i+1}. Skipping. Error: {e}")
                
        print("PROCESSING COMPLETE")


update_dataset_streaming("your-file.jsonl", "your-new-file.jsonl")