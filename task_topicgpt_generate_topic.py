import luigi
import pandas as pd
import os
from topicGPT.topicgpt_python import generate_topic_lvl1
import yaml

from task_topicgpt_extract_sample_generate_topic import TaskTopicGPTExtractSampleGenerateTopic

def insert_subdir_before_file(path, subdir):
    i = path.rfind("/")
    return path[:i] + "/" + subdir + path[i:]


class TaskTopicGPTGenerateTopic(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])
    model = luigi.Parameter(default=None)
    is_dump_sample = luigi.BoolParameter(default=True)

    def requires(self):
        return TaskTopicGPTExtractSampleGenerateTopic(self.domain, self.is_dump_sample)

    def run(self):

        with open("topicGPT/config.yml", "r") as f:
            config = yaml.safe_load(f)

        generate_topic_lvl1(
            "vllm",
            self.model,
            self.input().path,
            insert_subdir_before_file("topicGPT/"+config["generation"]["prompt"], self.domain),
            insert_subdir_before_file("topicGPT/"+config["generation"]["seed"], self.domain),
            self.output()["output"].path,
            self.output()["topic_output"].path,
            True
        )
        

    def output(self):
        model_path = self.model.split("/")[-1]
        base_dir = f'data/topicgpt/{self.domain}/{model_path}/update'

        os.makedirs(base_dir, exist_ok=True)

        dump_prefix = "dump_" if self.is_dump_sample else ""

        return {
            "output": luigi.LocalTarget(f'{base_dir}/generation_{dump_prefix}1.jsonl'),
            "topic_output": luigi.LocalTarget(f'{base_dir}/generation_{dump_prefix}1.md'),
        }
        
if __name__ == '__main__':
    tasks = [
        TaskTopicGPTGenerateTopic(domain=domain, model=model, is_dump_sample=True) for domain in ['restaurant', 'hotel'] for model in ["meta-llama/Llama-3.1-8B-Instruct"]
    ]
    luigi.build(tasks, local_scheduler=True)