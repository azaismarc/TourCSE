import luigi
import pandas as pd
import os
from topicGPT.topicgpt_python import refine_topics
import yaml

from task_topicgpt_generate_topic import TaskTopicGPTGenerateTopic

def insert_subdir_before_file(path, subdir):
    i = path.rfind("/")
    return path[:i] + "/" + subdir + path[i:]


class TaskTopicGPTRefineTopic(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])
    model = luigi.Parameter(default=None)

    def requires(self):
        return TaskTopicGPTGenerateTopic(self.domain, self.model)

    def run(self):

        with open("topicGPT/config.yml", "r") as f:
            config = yaml.safe_load(f)

        refine_topics(
            "vllm",
            self.model,
            "topicGPT/"+config["refinement"]["prompt"],
            self.input()['output'].path,
            self.input()['topic_output'].path,
            self.output()['topic_output'].path,
            self.output()['output'].path,
            verbose=True,
            remove=False,
            mapping_file=self.output()['mapping_file'].path,
        )
        

    def output(self):
        model_path = self.model.split("/")[-1]
        base_dir = f'data/topicgpt/{self.domain}/{model_path}/update'

        os.makedirs(base_dir, exist_ok=True)

        return {
            "output": luigi.LocalTarget(f'{base_dir}/refinement.jsonl'),
            "topic_output": luigi.LocalTarget(f'{base_dir}/refinement.md'),
            "mapping_file": luigi.LocalTarget(f'{base_dir}/refinement_mapping.json')
        }
        
if __name__ == '__main__':
    tasks = [
        TaskTopicGPTRefineTopic(domain=domain, model=model) for domain in ['restaurant', 'hotel'] for model in ["meta-llama/Llama-3.1-8B-Instruct"]
    ]
    luigi.build(tasks, local_scheduler=True)