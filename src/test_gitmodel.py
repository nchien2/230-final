import torch
import numpy as np
from transformers import AdamW, get_scheduler
from transformers import GitConfig, GitVisionConfig
from model_git import GitForCausalLM

# from model_image import GitForCausalLM

num_epochs = 1
batch_size = 2
num_frames = 3
visual_embed_size = 8576
sequence_length = 10
embed_hidden_size = 768

# # update the vision encoder embedding dim
vision_config = {"hidden_size": visual_embed_size}

config = GitConfig(vision_config)
print(config)

model = GitForCausalLM(config)
print(model.num_parameters())
optimizer = AdamW(model.parameters(), lr=3e-5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


visual_embed_v = np.load("1_baidu_soccer_embeddings.npy")
visual_embeds = torch.from_numpy(
    visual_embed_v[: batch_size * num_frames].reshape(batch_size, num_frames, -1)
)
print(visual_embed_v.shape)

model.train()
for epoch in range(num_epochs):
    batch = {  # "visual_embeds": torch.normal(0, 1, size=(25, 3, 768)),
        "visual_embeds": visual_embeds,
        "inputs_embeds": torch.normal(
            0, 1, size=(batch_size, sequence_length, embed_hidden_size)
        ),
    }
    # batch = {"pixel_values": torch.normal(0, 1, size=(2, 5, 3, 224, 224)),
    #          "inputs_embeds": torch.normal(0, 1, size=(2, 10, 768))}
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    print(outputs)
    # loss = outputs.loss
    # loss.backward()

    # optimizer.step()
    # optimizer.zero_grad()

    # break
