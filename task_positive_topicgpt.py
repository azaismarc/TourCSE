import luigi
import json
import csv
import random
import re
from collections import defaultdict
from tqdm import tqdm

from task_topicgpt_assignment_topic import TaskTopicGPTAssignmentTopic, TaskTopicGPTRefineTopic

topic_regex = re.compile(r'(?<=\]\s)([A-Za-z\s]+)(?=:)')

class TaskPositiveTopicGPT(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])
    model = luigi.Parameter(default=None)

    def requires(self):
        return {
            "assignment": TaskTopicGPTAssignmentTopic(domain=self.domain, model=self.model),
            "topic": TaskTopicGPTRefineTopic(self.domain, self.model)
        }
        
    def run(self):

        allowed_topics = set()
        with self.input()["topic"]["topic_output"].open('r') as f:
            for line in tqdm(f, desc="Reading allowed topics"):
                match = re.match(r'\[1\]\s([A-Za-z\s]+)\s\(Count:', line)
                if match:
                    allowed_topics.add(match.group(1).strip())
        
        data = defaultdict(list)

        # read jsonl
        with self.input()["assignment"].open('r') as f:
            for line in tqdm(f, desc="Processing assignment entries"):
                entry = json.loads(line.strip())
                _id = entry.get("id")
                responses = entry.get("responses", "")
                text = entry.get("text", "")
                rating = _id.split("-")[-1]
                
                # Extract all topics in this entry
                topics = topic_regex.findall(responses)
                
                for topic in topics:
                    t = topic.strip()
                    if t in allowed_topics:
                        data[topic.strip()+"-"+rating].append(text)     
                    else:
                        print(f"Not allowed :{t}")       
        
        datasets = []

        def pair(el, sets):
            if len(sets) == 1: 
                return (el, el)
            elements = [s for s in sets if s != el]
            return (el, random.choice(elements))

        for _, values in tqdm(data.items(), desc="Creating pairs"):
            sets = set(values)
            for el in sets:
                t1, t2 = pair(el, sets)
                datasets.append([t1, t2, 1])
            
        
        with self.output().open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text1', 'text2', 'label'])
            writer.writerows(datasets)

    def output(self):
        model_path = self.model.split("/")[-1]
        return luigi.LocalTarget(f'data/training/{self.domain}/topicgpt/{model_path}.tsv')
    
if __name__ == '__main__':
    tasks = [
        TaskPositiveTopicGPT(domain=domain, model="meta-llama/Llama-3.1-8B-Instruct") 
        for domain in ["hotel", "restaurant"]
    ]
    luigi.build(tasks, local_scheduler=True)
