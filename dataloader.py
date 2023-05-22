import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import os
import json
from SoccerNet.Evaluation.utils import FRAME_CLASS_DICTIONARY
from SoccerNet.utils import getListGames
import boto3
import io
import skvideo.io
import config

class SNV3Dataset(Dataset):
  def __init__(self, bucket='soccernet-230', split='train', resolution=(480,360)):

    # Initialize s3 resource
    s3 = boto3.resource('s3')
    self.session = boto3.Session(
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key,
    )
    self.s3 = session.resource('s3')
    # self.client = boto3.client('s3')
    # self.bucket = s3.Bucket(bucket)
    self.bucket = bucket

    # Get list of selected subset of games
    self.list_games = getListGames(split, task="frames")

    self.resolution = resolution
    self.resize = torchvision.transforms.Resize((resolution[1],resolution[0]), antialias=True)
    # self.preload_images = preload_images
    # self.zipped_images = zipped_images

    print("Reading the annotation files")
    self.metadata = list()
    for game in tqdm(self.list_games):
      # self.metadata.append(json.load(open(os.path.join(self.path, game, "Labels-v3.json"))))
      # print(f'getting key: {game}')
      try:
        obj_reference = self.s3.Object('soccernet-230', 'caption-2023/' + game + '/Labels-caption.json')
        obj = json.loads(obj_reference.get()['Body'].read().decode('utf-8'))
        obj['game'] = game
        self.metadata.append(obj)
      except:
        print(f'Cannot find game: {game}')
        continue

    self.data = list()
    for label in tqdm(self.metadata):
      # Retrieve each action in the game
      for annotation in label["annotations"]:
        # concatenate the replays of each action with itself
        annotation['game'] = label['game']
        self.data.append(annotation)

  def __getitem__(self, index):
    print('IN GET_ITEM')
    label = self.data[index]
    embed_ref = self.s3.Object('soccernet-230', 'caption-2023/' + label['game'] + '/1_baidu_soccer_embeddings.npy')
    vid_ref = self.s3.Object('soccernet-230', 'videos/' + label['game'] + '/1_224p.mkv')

    with io.BytesIO(embed_ref.get()["Body"].read()) as e_f:
      e_f.seek(0)  # rewind the file
      embed = np.load(e_f, allow_pickle=True)
      print(embed)
    with io.BytesIO(vid_ref.get()["Body"].read()) as v_f:
      print('LOADING VIDEO')
      v_f.seek(0)  # rewind the file
      video = mpimg.imread(v_f)
      print(video)

    out = {
      'label' : label,
      'embedding' : embed,
      'video' : video
    }
    return out

if __name__ == "__main__":
    session = boto3.Session(
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key,
    )
    s3 = session.resource('s3')
    test_ds = SNV3Dataset(split='test',
                      bucket='soccernet-230')
    for example in test_ds:
      print(example)
      break
