import torch

_ALLTRACKER_URL = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"

dependencies = ["torch"]


def alltracker(*, pretrained: bool = True, **kwargs):
    """
    AllTracker: Efficient Dense Point Tracking at High Resolution.
    Harley et al., ICCV 2025. https://github.com/aharley/alltracker
    """
    from nets.alltracker import Net

    model = Net(seqlen=16, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_ALLTRACKER_URL, map_location="cpu")
        model.load_state_dict(state_dict["model"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
