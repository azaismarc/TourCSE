import luigi
import pandas as pd
import json
import nltk

from task_extract_sample import TaskExtractSample

class TaskTopicGPTExtractSampleAssignmentTopic(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel", "sentiment"])

    def requires(self):
        if self.domain in ["restaurant", "hotel"]:
            return TaskExtractSample(self.domain)

    def run(self):

        if self.domain in ["restaurant", "hotel"]:

            df = pd.read_csv(self.input().path, delimiter='\t')

            # df = df[df['split'] == 'train']

            # If you want lists
            reviews = df['review'].tolist()
            ratings = df['rating'].tolist()

            data = []

            for r_id, (rating, review) in enumerate(zip(ratings, reviews)):
                for s_id, sent in enumerate(nltk.sent_tokenize(review)):
                    if len(sent.split()) < 3:
                        continue
                    _id = f"{r_id}-{s_id}-{rating}"
                    data.append({
                        "id": _id,
                        "text": sent
                    })

        else:

            df = pd.read_csv("data/sentiment_samples.tsv", sep="\t", index_col="_id")

            sentences = df['sentence'].tolist()
            data = []
            # Iterate over your dataframe
            for r_id, sentence in zip(df.index, df['sentence']):
                # Tokenize sentence into smaller sentences if needed
                for s_id, sent in enumerate(nltk.sent_tokenize(sentence)):
                    if len(sent.split()) < 3:
                        continue
                    _id = f"{r_id}"  # use df index + sentence id
                    data.append({
                        "id": _id,
                        "text": sent
                    })

        with self.output().open('w') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def output(self):
        return luigi.LocalTarget(f'data/topicgpt/{self.domain}/time_10_000/assignment_sample.jsonl')
    
if __name__ == '__main__':
    tasks = [
        TaskTopicGPTExtractSampleAssignmentTopic(domain=domain) for domain in ['restaurant', 'hotel']
    ]
    luigi.build(tasks, local_scheduler=True)