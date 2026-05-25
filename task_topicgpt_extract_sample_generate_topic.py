import luigi
import pandas as pd
import json

from task_extract_sample import TaskExtractSample

class TaskTopicGPTExtractSampleGenerateTopic(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])
    is_dump_sample = luigi.BoolParameter(default=False)
    N = luigi.Parameter(default=200)
    


    def requires(self):
        if not self.is_dump_sample:
            return TaskExtractSample(self.domain)

    def run(self):

        if not self.is_dump_sample:

            df = pd.read_csv(self.input().path, delimiter='\t')

            df = df[df['split'] == 'train']

            df = df.groupby('rating').sample(n=self.N, random_state=42)

            # If you want lists
            reviews = df['review'].tolist()
            
            reviews = df['review'].tolist()
            data = [{"text": r} for r in reviews]
        
        else:
            data = [{"text": ""}]

        # Save as JSONL
        with self.output().open('w') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def output(self):
        return luigi.LocalTarget(
            f'data/topicgpt/{self.domain}/generation_{"dump_" if self.is_dump_sample else ""}sample.jsonl'
        )    
if __name__ == '__main__':
    tasks = [
        TaskTopicGPTExtractSampleGenerateTopic(domain=domain, is_dump_sample=True) for domain in ['restaurant', 'hotel']
    ]
    luigi.build(tasks, local_scheduler=True)