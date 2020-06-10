import logging
import s3fs
import boto3
import json
import pandas as pd
import os

from model.opts import configure_args
from model.utils import set_logger


# detect the dominant language of a sentence
def detect_language(note):
    if len(note.encode('utf-8')) < 5000:
        comprehend = boto3.client(service_name='comprehend', region_name='us-east-2')
        lang = comprehend.detect_dominant_language(Text = note)
        return lang['Languages'][0]['LanguageCode']
    else:
        return None


def detect_sentiment(note):
    global pos_count
    global neg_count
    comprehend = boto3.client(service_name='comprehend', region_name='us-east-2')
    sentiment = comprehend.detect_sentiment(Text=note, LanguageCode='en')
    if sentiment['Sentiment'] == 'POSITIVE':
        pos_count += 1
        return 1
    else:
        neg_count += 1
        return 0


if __name__ == "__main__":
    args = configure_args()

    if not os.path.exists('output'):
        os.makedirs('output')

    if not os.path.exists('data'):
        os.makedirs('data')

    set_logger('output/train.log')

    fs = s3fs.S3FileSystem()

    bucket_name = 'cureskin-dataset'
    data_key = 'dr_msg_stats.csv'
    data_location = 's3://{}/{}'.format(bucket_name, data_key)

    df_stats = pd.read_csv(data_location)

    s3 = boto3.resource('s3')
    bucket = s3.Bucket('cureskin-dataset')
    df = pd.DataFrame()
    # extract users with doctor's notes
    for user in df_stats['user_id'].unique()[:args.data_size]:
        data_location = 's3://cureskin-dataset/followup_data/user_{0:012}.json'.format(user)
        df_key = pd.read_json(data_location)
        if df.empty:
            df = df_key
        else:
            df = df.append(df_key, ignore_index=True, sort=False)

    df_doctor = df[df['entry_type'] == 'DrMessage']
    df_doctor['language'] = df_doctor['dr_note'].apply(detect_language)
    df_doctor = df_doctor[df_doctor['language'] == 'en']

    df_annotate = df[df['entry_type'] == 'Annotation']

    total = []
    pos_count = 0
    neg_count = 0
    user_count = 0
    # only extract doctor's notes and user's images
    with open('data/user_data.txt', 'w') as note_file:
        for user in df_doctor['userId'].unique()[:args.data_size]:
            df_user = df_annotate[df_annotate['userId'] == user].sort_values(by='created_at')
            user_img = df_user['image_path'].tolist()
            total.append(len(user_img))
            if 5 < len(user_img) < args.frame_size:
                df_user_note = df_doctor[df_doctor['userId'] == user].sort_values(by='created_at')
                user_note = str(df_user_note['dr_note'].tolist()[-1])
                sent = detect_sentiment(user_note)

                json.dump([str(user)] + [str(sent)] + [str(user_note)] + user_img, note_file)
                note_file.write('\n')
                user_count += 1

    logging.info('Average number of images per user: {}'.format(sum(total)/len(total)))
    logging.info('Minimum number of images per user: {}'.format(min(total)))
    logging.info('Maximum number of images per user: {}'.format(max(total)))
    logging.info('Number of positive examples (1): {}'.format(pos_count))
    logging.info('Number of negative examples (0): {}'.format(neg_count))
    logging.info('Number of valid users in dataset: {}'.format(user_count))
