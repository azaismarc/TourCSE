import luigi
import os
from topicGPT.topicgpt_python import assign_topics
import yaml

from task_topicgpt_refine_topic import TaskTopicGPTRefineTopic
from task_topicgpt_extract_sample_assignment_topic import TaskTopicGPTExtractSampleAssignmentTopic

class TaskTopicGPTAssignmentTopic(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel", "sentiment"])
    model = luigi.Parameter(default="Llama")

    def requires(self):
        d = dict()
        if self.domain in ["hotel", "restaurant"]:
            d["topic"] = TaskTopicGPTRefineTopic(self.domain, self.model)
        d["sample"] = TaskTopicGPTExtractSampleAssignmentTopic(self.domain)
        return d

    def run(self):

        with open("topicGPT/config.yml", "r") as f:
            config = yaml.safe_load(f)
        
        if self.domain in ["hotel", "restaurant"]:

            assign_topics(
                "vllm",
                self.model,
                self.input()['sample'].path,
                "topicGPT/"+config["assignment"]["prompt"],
                self.output().path,
                self.input()['topic']['topic_output'].path,
                True
            )
        
        else:

            assign_topics(
                "vllm",
                self.model,
                self.input()['sample'].path,
                "topicGPT/prompt/assignment_sentiment.txt",
                self.output().path,
                "data/topicgpt/sentiment.md",
                True
            )

        

    def output(self):
        model_path = self.model.split("/")[-1]
        folder_path = f'data/topicgpt/{self.domain}/{model_path}'

        # Crée le dossier s'il n'existe pas déjà
        os.makedirs(folder_path, exist_ok=True)

        return luigi.LocalTarget(f'{folder_path}/assignment.jsonl')
        
    
if __name__ == '__main__':
    tasks = [
        TaskTopicGPTAssignmentTopic(domain=domain, model=model) for domain in ['sentiment'] for model in ["meta-llama/Llama-3.1-8B-Instruct"]
    ]
    luigi.build(tasks, local_scheduler=True)