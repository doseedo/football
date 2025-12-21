# Player Re-Identification System

A complete unsupervised player re-identification system that can identify players without prior training on the match, matching SkillCorner's capabilities.

## Components

### 1. OSNet (Omni-Scale Network)
**File**: `osnet.py`

Extracts 512-dimensional appearance embeddings from player crops using the OSNet-AIN architecture.

**Features**:
- Omni-scale feature learning with multiple receptive fields
- Channel attention mechanism (AIN)
- Pretrained on person re-ID datasets
- L2-normalized embeddings for similarity matching

**Pretrained Weights**:
Download from [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid):

```bash
# Recommended model (best performance)
wget https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.0/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth \
     -O models/osnet_ain_x1_0.pth

# Alternative (smaller, faster)
wget https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.0/osnet_ain_x1_0_market1501_256x128_amsgrad_ep100_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth \
     -O models/osnet_ain_market.pth
```

**Usage**:
```python
from identity import ReIDExtractor

# Initialize
extractor = ReIDExtractor(model_path='models/osnet_ain_x1_0.pth', device='cuda')

# Extract features
embeddings = extractor.extract_features(player_crops)  # (B, 512)

# Compute similarity
similarity = extractor.compute_similarity(embeddings1, embeddings2)
```

### 2. Jersey Number Detector
**File**: `jersey_detector.py`

Localizes jersey number regions on player crops using spatial attention.

**Features**:
- Attention-based number localization
- Handles front and back numbers
- Color-based refinement
- **Heuristic fallback** when model not available

**Training** (optional):
The detector includes a heuristic method that works well without training. For better performance, train on annotated football images:

```python
from identity import JerseyNumberDetector

# Train on your dataset with bbox annotations
model = JerseyNumberDetector()
# ... training code ...
```

**Usage**:
```python
from identity import JerseyDetector

# Initialize (works without pretrained weights using heuristic)
detector = JerseyDetector(use_heuristic=True)

# Detect number regions
results = detector.detect(player_crops)
# Returns: [{'bbox': [x1,y1,x2,y2], 'confidence': float, 'crop': ndarray}]
```

### 3. Jersey Number Recognizer
**File**: `jersey_recognizer.py`

Recognizes jersey numbers (1-99) using CRNN architecture with CTC loss.

**Features**:
- CNN + BiLSTM + CTC pipeline
- Spatial Transformer Network for alignment
- Handles partial occlusions
- Temporal aggregation for robust predictions

**Pretrained Weights**:
Train on synthetic jersey number dataset or use text recognition models:

```bash
# Option 1: Train on synthetic data
python scripts/train_jersey_recognizer.py --data synthetic_jerseys/

# Option 2: Fine-tune from text recognition model
# Download CRNN pretrained on text
wget https://www.example.com/crnn_text_recognition.pth -O models/crnn_base.pth
python scripts/finetune_jersey_recognizer.py --init models/crnn_base.pth
```

**Usage**:
```python
from identity import JerseyRecognizer, TemporalNumberAggregator

# Initialize
recognizer = JerseyRecognizer(model_path='models/jersey_recognizer.pth')
aggregator = TemporalNumberAggregator(window_size=30)

# Recognize numbers
results = recognizer.recognize(number_crops)
# Returns: [{'number': int, 'text': str, 'confidence': float}]

# Temporal aggregation
for track_id, result in zip(track_ids, results):
    aggregator.add_prediction(track_id, result['number'], result['confidence'])
    stable_number = aggregator.get_stable_number(track_id, min_votes=3)
```

### 4. Unsupervised Team Classifier
**File**: `contrastive_team.py`

Discovers team clusters without labels using color-based features and contrastive learning.

**Features**:
- Color histogram extraction from jersey regions
- K-means or DBSCAN clustering
- Online updates as new players appear
- Automatic referee detection

**No pretrained weights needed** - completely unsupervised!

**Usage**:
```python
from identity import OnlineTeamClassifier

# Initialize
classifier = OnlineTeamClassifier(num_teams=3, update_frequency=30)

# Process frames
for frame_crops, track_ids in video:
    team_ids = classifier.update(frame_crops, track_ids)

# Get team info
info = classifier.get_team_info()
# Returns: {'team_colors': {...}, 'referee_id': int}
```

### 5. Combined Player Identifier
**File**: `player_identifier.py`

Fuses all components for robust player identification.

**Features**:
- Multi-modal fusion (appearance + number + team)
- Player gallery maintenance
- Track-to-player matching
- Temporal consistency
- Stable player IDs across the match

**Usage**:
```python
from identity import create_player_identifier

# Initialize with all components
identifier = create_player_identifier({
    'osnet_path': 'models/osnet_ain_x1_0.pth',
    'jersey_recognizer_path': 'models/jersey_recognizer.pth',
    'device': 'cuda',
    'num_teams': 3
})

# Process frame
identities = identifier.process_frame(player_crops, track_ids)

# Access results
for identity in identities:
    print(f"Player {identity.stable_id}")
    print(f"  Team: {identity.team_id}")
    print(f"  Jersey: #{identity.jersey_number}")
    print(f"  Confidence: {identity.confidence:.2f}")

# Query specific player
stable_id = identifier.get_stable_id(track_id)
jersey_num = identifier.get_jersey_number(track_id)

# Export match roster
roster = identifier.export_roster()
```

## Complete Pipeline Example

```python
import cv2
import numpy as np
from identity import create_player_identifier, visualize_identity

# Initialize
identifier = create_player_identifier({
    'osnet_path': 'models/osnet_ain_x1_0.pth',
    'device': 'cuda'
})

# Process video
cap = cv2.VideoCapture('match.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get player detections from your detector/tracker
    player_crops, track_ids = detect_and_track(frame)

    # Identify players
    identities = identifier.process_frame(player_crops, track_ids)

    # Visualize
    for identity, crop in zip(identities, player_crops):
        vis = visualize_identity(crop, identity)
        cv2.imshow(f'Player {identity.stable_id}', vis)

# Export final roster
roster = identifier.export_roster()
print(f"Identified {len(roster)} players")

for player_id, info in roster.items():
    print(f"Player {player_id}: Team {info['team_id']}, #{info['jersey_number']}")
```

## Model Downloads

### Required (OSNet):
```bash
mkdir -p models/
cd models/

# OSNet-AIN pretrained on MSMT17 (recommended)
wget https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.0/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth \
     -O osnet_ain_x1_0.pth
```

### Optional (Jersey Recognizer):
Train on your own dataset or use provided synthetic data generator:

```bash
# Generate synthetic jersey numbers
python scripts/generate_synthetic_jerseys.py --output data/synthetic_jerseys/ --num-samples 10000

# Train recognizer
python scripts/train_jersey_recognizer.py \
    --data data/synthetic_jerseys/ \
    --epochs 50 \
    --batch-size 32 \
    --output models/jersey_recognizer.pth
```

## Performance Notes

### OSNet Pretrained Weights
- **MSMT17**: Best generalization, recommended for football
- **Market1501**: Smaller model, faster inference
- **DukeMTMC**: Good alternative

All models work well for sports re-identification without fine-tuning.

### Jersey Numbers
- Visible ~30% of frames
- Temporal aggregation essential (30-frame window)
- Minimum 3 votes for stable assignment

### Team Classification
- Works completely unsupervised
- Accuracy improves over time (online learning)
- Handles 2 teams + referee automatically

### Matching Strategy
1. **Jersey number** (if available): Highest priority
2. **Appearance** (OSNet): Similarity threshold = 0.7
3. **Team**: Must match for re-identification
4. **Temporal consistency**: Track history voting

## Key Insights

1. **OSNet pretrained weights**: Pre-trained person re-ID models work excellently for football players without fine-tuning.

2. **Jersey number visibility**: Numbers are only visible in ~30% of frames. Temporal aggregation with voting is critical.

3. **Team colors**: More reliable than individual appearance. Color-based clustering is surprisingly effective.

4. **Contrastive learning**: Discovers teams without labels by maximizing inter-team distance.

5. **Multi-modal fusion**: Combining all three cues (appearance, number, team) provides robust identification.

## Architecture Summary

```
Input: Player Crops (from detector/tracker)
    ↓
┌───────────────────────────────────────────────┐
│  Parallel Feature Extraction                  │
├─────────────────┬─────────────┬───────────────┤
│  OSNet          │  Jersey     │  Color        │
│  Appearance     │  Number     │  Features     │
│  (512-dim)      │  (1-99)     │  (RGB/HSV)    │
└─────────────────┴─────────────┴───────────────┘
    ↓                   ↓               ↓
┌─────────────────┬─────────────┬───────────────┐
│  Embedding      │  OCR        │  Clustering   │
│  Matching       │  +Voting    │  (K-means)    │
└─────────────────┴─────────────┴───────────────┘
    ↓                   ↓               ↓
┌───────────────────────────────────────────────┐
│  Player Identity Fusion                       │
│  - Gallery matching                           │
│  - Temporal consistency                       │
│  - Confidence weighting                       │
└───────────────────────────────────────────────┘
    ↓
Output: PlayerIdentity
  - stable_id: Consistent player ID
  - team_id: Team assignment
  - jersey_number: Jersey number
  - confidence: Overall confidence
```

## Citation

If you use this implementation, please cite:

```bibtex
@inproceedings{zhou2019omni,
  title={Omni-Scale Feature Learning for Person Re-Identification},
  author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
  booktitle={ICCV},
  year={2019}
}
```
