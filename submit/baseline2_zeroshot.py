import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
from types import SimpleNamespace

from baseline1_fewshot import UCF101Dataset, collate_fn, extract_features

UCF101_CLASSES = [
    "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
    "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
    "Biking", "Billiards", "BlowDryHair", "BlowingCandles", "BodyWeightSquats",
    "Bowling", "BoxingPunchingBag", "BoxingSpeedBag", "BreastStroke", "BrushingTeeth",
    "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot", "CuttingInKitchen",
    "Diving", "Drumming", "Fencing", "FieldHockeyPenalty", "FloorGymnastics",
    "FrisbeeCatch", "FrontCrawl", "GolfSwing", "Haircut", "Hammering",
    "HammerThrow", "HandstandPushups", "HandstandWalking", "HeadMassage", "HighJump",
    "HorseRace", "HorseRiding", "HulaHoop", "IceDancing", "JavelinThrow",
    "JugglingBalls", "JumpingJack", "JumpRope", "Kayaking", "Knitting",
    "LongJump", "Lunges", "MilitaryParade", "Mixing", "MoppingFloor",
    "Nunchucks", "ParallelBars", "PizzaTossing", "PlayingCello", "PlayingDaf",
    "PlayingDhol", "PlayingFlute", "PlayingGuitar", "PlayingPiano", "PlayingSitar",
    "PlayingTabla", "PlayingViolin", "PoleVault", "PommelHorse", "PullUps",
    "Punch", "PushUps", "Rafting", "RockClimbingIndoor", "RopeClimbing",
    "Rowing", "SalsaSpin", "ShavingBeard", "Shotput", "SkateBoarding",
    "Skiing", "Skijet", "SkyDiving", "SoccerJuggling", "SoccerPenalty",
    "StillRings", "SumoWrestling", "Surfing", "Swing", "TableTennisShot",
    "TaiChi", "TennisSwing", "ThrowDiscus", "TrampolineJumping", "Typing",
    "UnevenBars", "VolleyballSpiking", "WalkingWithDog", "WallPushups", "WritingOnBoard",
    "YoYo"
]


def create_text_prompts(class_names):
    templates = [
        "a video of a person {}",
        "a person {}",
        "someone {}",
        "a video showing {}",
        "{}"
    ]
    all_prompts = []
    for class_name in class_names:
        readable = ''.join([' ' + c.lower() if c.isupper() else c for c in class_name]).strip()
        all_prompts.append([t.format(readable) for t in templates])
    return all_prompts


def extract_text_features(model, processor, prompts, device):
    model.eval()
    class_features = []

    with torch.no_grad():
        for class_prompts in tqdm(prompts):
            prompt_features = []
            for prompt in class_prompts:
                inputs = processor(text=[prompt], return_tensors="pt", padding=True)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)

                if isinstance(model, torch.nn.DataParallel):
                    text_outputs = model.module.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    text_outputs = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

                prompt_features.append(F.normalize(text_outputs, dim=-1))

            class_feature = F.normalize(torch.stack(prompt_features).mean(dim=0), dim=-1)
            class_features.append(class_feature)

    return torch.cat(class_features, dim=0)


def zero_shot_classify(video_features, text_features, temperature=1.0):
    similarity = video_features @ text_features.T
    logits = similarity / temperature
    return logits.argmax(dim=-1), F.softmax(logits, dim=-1)


def evaluate_zero_shot(video_features, text_features, labels, class_names):
    device = text_features.device
    video_features = video_features.to(device)
    labels = labels.to(device)

    predictions, probabilities = zero_shot_classify(video_features, text_features)

    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    probabilities = probabilities.cpu().numpy()

    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro')

    top5_preds = probabilities.argsort(axis=-1)[:, -5:]
    top5_accuracy = np.mean([label in top5 for label, top5 in zip(labels, top5_preds)])

    return {
        'accuracy': float(accuracy),
        'top5_accuracy': float(top5_accuracy),
        'macro_f1': float(macro_f1)
    }, predictions, probabilities


def main():
    args = SimpleNamespace(
        model_name="microsoft/xclip-base-patch16",
        cache_dir="/mnt/amlfs-03/shared/ayaanm/yanav/cache",
        num_frames=32,
        input_size=224,
        scale_resize=256,
        batch_size=8,
        num_workers=4,
        use_multi_gpu=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    test_dataset = UCF101Dataset(
        split='test', processor=processor, num_frames=args.num_frames,
        input_size=args.input_size, scale_resize=args.scale_resize,
        use_train_augmentation=False, cache_dir=args.cache_dir
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )

    prompts = create_text_prompts(UCF101_CLASSES)
    text_features = extract_text_features(model, processor, prompts, device)

    test_indices = list(range(len(test_dataset)))
    video_features, labels = extract_features(
        model, processor, test_loader, device,
        use_multi_gpu=args.use_multi_gpu, args=args,
        dataset=test_dataset, indices=test_indices
    )
    video_features = torch.from_numpy(video_features)
    labels = torch.from_numpy(labels)

    results, _, _ = evaluate_zero_shot(video_features, text_features, labels, UCF101_CLASSES)
    return results


if __name__ == "__main__":
    main()

