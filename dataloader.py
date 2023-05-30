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
import random
from collections import Counter
from torchtext.vocab import vocab


class SoccerNetVideoProcessor(object):
    """video_fn is a tuple of (video_id, half, frame)."""

    def __init__(self, clip_length):
        self.clip_length = clip_length

    def __call__(self, video_fn, feats):
        video_id, half, frame = video_fn
        video_feature = feats[video_id][half]
        #make sure that the clip lenght is right
        start = min(frame, video_feature.shape[0] - self.clip_length)
        video_feature = video_feature[start : start + self.clip_length]

        return video_feature


class SoccerNetTextProcessor(object):
    """
    A generic Text processor
    tokenize a string of text on-the-fly.
    """

    def __init__(self, corpus, min_freq=5, load_path=None, split='train'):
        import spacy
        self.split = split
        self.load_path = load_path

        spacy_token = spacy.load("en_core_web_sm").tokenizer
        # Add special case rule
        spacy_token.add_special_case("[PLAYER]", [{"ORTH": "[PLAYER]"}])
        spacy_token.add_special_case("[COACH]", [{"ORTH": "[COACH]"}])
        spacy_token.add_special_case("[TEAM]", [{"ORTH": "[TEAM]"}])
        spacy_token.add_special_case("([TEAM])", [{"ORTH": "([TEAM])"}])
        spacy_token.add_special_case("[REFEREE]", [{"ORTH": "[REFEREE]"}])
        self.tokenizer = lambda s: [c.text for c in spacy_token(s)]
        self.min_freq = min_freq

        if self.load_path:
          self.vocab = torch.load(load_path)
        else:
          self.build_vocab(corpus)


    def build_vocab(self, corpus):
        counter = Counter([token for c in corpus for token in self.tokenizer(c)])
        voc = vocab(counter, min_freq=self.min_freq, specials=["[PAD]", "[SOS]", "[EOS]", "[UNK]", "[MASK]", "[CLS]"])
        voc.set_default_index(voc['[UNK]'])
        torch.save(voc, f'{self.split}_vocab.pyi')
        self.vocab = voc

    def __call__(self, text):
        return self.vocab(self.tokenizer(text))

    def detokenize(self, tokens):
        return " ".join(self.vocab.lookup_tokens(tokens))


class SNV3Dataset(Dataset):
  def __init__(self,
               bucket='soccernet-230',
               split='train',
               resolution=(480,360),
               framerate=2,
               window_size=15,
               batch_size=32,
               include_vid=False,
               vocab_path=None):

    self.batch_size = batch_size
    self.framerate = framerate
    self.include_vid = include_vid
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
    self.clip_len = window_size * framerate

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

    #launch a VideoProcessor that will create a clip around a caption
    self.video_processor = SoccerNetVideoProcessor(self.clip_len)
    #launch a TextProcessor that will tokenize a caption
    self.text_processor = SoccerNetTextProcessor(self.getCorpus(split=[split]),
                                                 split=split,
                                                 load_path=vocab_path)
    self.vocab_size = len(self.text_processor.vocab)

  def __len__(self):
    return len(self.list_games)

  def ref_to_tensor(self, ref):
    with io.BytesIO(ref.get()["Body"].read()) as f:
      f.seek(0)  # rewind the file
      np_array = np.load(f, allow_pickle=True)
      out = torch.from_numpy(np_array)
      return out

  def __getitem__(self, index):
    print('IN GET_ITEM')
    game = self.list_games[index]
    label_reference = self.s3.Object('soccernet-230', 'caption-2023/' + game + '/Labels-caption.json')
    label = json.loads(label_reference.get()['Body'].read().decode('utf-8'))
    captions = random.sample(label['annotations'], self.batch_size)
    pad_len = max([len(x['anonymized']) for x in captions])

    emb_ref1 = self.s3.Object('soccernet-230', 'caption-2023/' + game + '/1_baidu_soccer_embeddings.npy')
    emb_ref2 = self.s3.Object('soccernet-230', 'caption-2023/' + game + '/2_baidu_soccer_embeddings.npy')

    embed_list = [self.ref_to_tensor(emb_ref1), self.ref_to_tensor(emb_ref2)]
    embed = torch.cat(embed_list)

    out = {
        'embed': [],
        'caption': [],
        'video': []
    }
    for annotation in captions:
      time = annotation["gameTime"]
      event = annotation["label"]
      half = int(time[0])
      if half > 2:
          continue

      minutes, seconds = time.split(' ')[-1].split(':')
      minutes, seconds = int(minutes), int(seconds)
      frame = self.framerate * ( seconds + 60 * minutes)

      start = min(frame, embed.shape[0] - self.clip_len)
      emb_clip = embed[start:start + self.clip_len,:]

      video = None
      if self.include_vid:
        vid_ref1 = self.s3.Object('soccernet-230', 'videos/' + game + '/1_224p.mkv')
        vid_ref2 = self.s3.Object('soccernet-230', 'videos/' + game + '/2_224p.mkv')
        with io.BytesIO(vid_ref.get()["Body"].read()) as v_f:
          v_f.seek(0)  # rewind the file
          video = v_f.read()

          idxs = range(0, 10)
          video_list = [imageio.v3.imread(video, format_hint='.mkv', index=idx, plugin='pyav') for idx in idxs]
          video = np.stack(video_list)
          print(video.shape)

      caption_tokens = self.text_processor(annotation['anonymized'])
      caption_tokens += [0] * (pad_len - len(caption_tokens))

      out['embed'].append(emb_clip)
      out['caption'].append(torch.tensor(caption_tokens))
      out['video'].append(video)

    out['embed'] = torch.stack(out['embed'])
    out['caption'] = torch.stack(out['caption'])
    if self.include_vid:
      out['video'] = torch.stack(out['video'])

    return out

  def getCorpus(self, split=["train"]):
    """
    Args:
        split (string): split of dataset
    Returns:
        corpus (List[string]): vocabulary build from split.
    """
    corpus = [annotation['anonymized'] for game in getListGames(split, task="caption") for annotation in self.data]
    return corpus

  def detokenize(self, tokens, remove_EOS=True):
    """
    Args:
        tokens (List[int]): tokens of caption
    Returns:
        caption (string): string obtained after replacing each token by its corresponding word
    """
    string = self.text_processor.detokenize(tokens)
    return string.rstrip(f" {self.text_processor.vocab.lookup_token(config.EOS_TOKEN)}") if remove_EOS else string


if __name__ == "__main__":
    session = boto3.Session(
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key,
    )
    s3 = session.resource('s3')
    test_ds = SNV3Dataset(split='test',
                      bucket='soccernet-230',
                      vocab_path='vocab_files/test_vocab.pyi')
    for example in test_ds:
      print(example)
      break
