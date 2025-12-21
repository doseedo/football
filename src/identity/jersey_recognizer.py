"""Jersey number recognizer using CRNN (CNN + BiLSTM + CTC).

Recognizes jersey numbers (1-99) from cropped number regions.
Handles partially visible and occluded numbers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import cv2
from collections import Counter


class SpatialTransformerNetwork(nn.Module):
    """Spatial Transformer Network for aligning number regions."""

    def __init__(self):
        super().__init__()

        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3x2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0],
                                                     dtype=torch.float))

    def forward(self, x):
        """
        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Aligned images (B, 3, H, W)
        """
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x


class CRNN(nn.Module):
    """CRNN architecture for jersey number recognition.

    Architecture:
        1. CNN: Extract visual features
        2. BiLSTM: Sequence modeling
        3. CTC: Connectionist Temporal Classification for alignment-free recognition
    """

    def __init__(self, num_classes: int = 11, hidden_size: int = 256,
                 use_stn: bool = True):
        """
        Args:
            num_classes: Number of character classes (10 digits + blank)
            hidden_size: Hidden size of LSTM
            use_stn: Whether to use Spatial Transformer Network
        """
        super().__init__()

        self.use_stn = use_stn
        if use_stn:
            self.stn = SpatialTransformerNetwork()

        # CNN backbone for feature extraction
        # Input: (B, 3, 32, 100)
        self.cnn = nn.Sequential(
            # Conv layer 1
            nn.Conv2d(3, 64, 3, 1, 1),  # -> (B, 64, 32, 100)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # -> (B, 64, 16, 50)

            # Conv layer 2
            nn.Conv2d(64, 128, 3, 1, 1),  # -> (B, 128, 16, 50)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # -> (B, 128, 8, 25)

            # Conv layer 3
            nn.Conv2d(128, 256, 3, 1, 1),  # -> (B, 256, 8, 25)
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Conv layer 4
            nn.Conv2d(256, 256, 3, 1, 1),  # -> (B, 256, 8, 25)
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # -> (B, 256, 4, 25)

            # Conv layer 5
            nn.Conv2d(256, 512, 3, 1, 1),  # -> (B, 512, 4, 25)
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # Conv layer 6
            nn.Conv2d(512, 512, 3, 1, 1),  # -> (B, 512, 4, 25)
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # -> (B, 512, 2, 25)

            # Conv layer 7
            nn.Conv2d(512, 512, 2, 1, 0),  # -> (B, 512, 1, 24)
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        # Recurrent layers
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Log probabilities (T, B, num_classes) where T is sequence length
        """
        # Spatial transformation
        if self.use_stn:
            x = self.stn(x)

        # CNN features
        conv = self.cnn(x)  # (B, 512, 1, W)

        # Prepare for RNN: (B, C, H, W) -> (W, B, C*H)
        b, c, h, w = conv.size()
        assert h == 1, "Height of conv output must be 1"
        conv = conv.squeeze(2)  # (B, C, W)
        conv = conv.permute(2, 0, 1)  # (W, B, C)

        # RNN
        output = self.rnn(conv)  # (W, B, num_classes)

        # Log softmax for CTC loss
        output = F.log_softmax(output, dim=2)

        return output


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM layer."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True,
                          batch_first=False)
        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        Args:
            x: Input (T, B, input_size)

        Returns:
            Output (T, B, output_size)
        """
        recurrent, _ = self.rnn(x)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # (T * b, output_size)
        output = output.view(T, b, -1)

        return output


class CTCDecoder:
    """CTC decoder for converting network output to text."""

    def __init__(self, characters: str = '0123456789'):
        """
        Args:
            characters: Character set (digits 0-9)
        """
        self.characters = characters
        self.num_classes = len(characters) + 1  # +1 for CTC blank
        self.blank_idx = len(characters)

    def decode(self, probs: torch.Tensor, method: str = 'greedy') -> List[str]:
        """
        Decode CTC output to text.

        Args:
            probs: Log probabilities (T, B, num_classes)
            method: Decoding method ('greedy' or 'beam_search')

        Returns:
            List of decoded strings
        """
        if method == 'greedy':
            return self._greedy_decode(probs)
        elif method == 'beam_search':
            return self._beam_search_decode(probs)
        else:
            raise ValueError(f"Unknown decoding method: {method}")

    def _greedy_decode(self, probs: torch.Tensor) -> List[str]:
        """Greedy decoding (argmax at each timestep)."""
        _, preds = probs.max(2)  # (T, B)
        preds = preds.transpose(1, 0).contiguous()  # (B, T)

        results = []
        for pred in preds:
            # Remove consecutive duplicates and blank
            char_list = []
            prev_char = None
            for p in pred:
                p = int(p)
                if p != self.blank_idx and p != prev_char:
                    char_list.append(self.characters[p])
                prev_char = p

            results.append(''.join(char_list))

        return results

    def _beam_search_decode(self, probs: torch.Tensor, beam_width: int = 5) -> List[str]:
        """Beam search decoding (simple version)."""
        # For simplicity, use greedy for now
        # Full beam search implementation is more complex
        return self._greedy_decode(probs)


class JerseyRecognizer:
    """High-level interface for jersey number recognition."""

    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        """
        Args:
            model_path: Path to trained CRNN model
            device: Device to run on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = CRNN(num_classes=11, hidden_size=256, use_stn=True)

        # Load weights if provided
        if model_path:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded jersey recognizer from {model_path}")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Decoder
        self.decoder = CTCDecoder()

        # Input size
        self.img_h = 32
        self.img_w = 100

    def preprocess(self, images: np.ndarray) -> torch.Tensor:
        """
        Preprocess images for CRNN.

        Args:
            images: Images (B, H, W, 3) in [0, 255] range

        Returns:
            Preprocessed tensor (B, 3, 32, 100)
        """
        batch = []
        for img in images:
            # Resize to fixed size
            img = cv2.resize(img, (self.img_w, self.img_h))

            # Convert to grayscale and back to RGB (for consistency)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            # Normalize
            img = img.astype(np.float32) / 255.0

            batch.append(img)

        batch = np.array(batch)
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2)  # (B, 3, H, W)

        return batch

    @torch.no_grad()
    def recognize(self, images: np.ndarray, return_confidence: bool = True) -> List[dict]:
        """
        Recognize jersey numbers from cropped number regions.

        Args:
            images: Number region crops (B, H, W, 3) or (H, W, 3)
            return_confidence: Whether to return confidence scores

        Returns:
            List of dicts with keys:
                - number: Recognized number as integer (or None if invalid)
                - text: Raw text output
                - confidence: Recognition confidence [0, 1]
        """
        # Handle single image
        if images.ndim == 3:
            images = images[np.newaxis]

        # Preprocess
        x = self.preprocess(images)
        x = x.to(self.device)

        # Recognize
        probs = self.model(x)  # (T, B, num_classes)

        # Decode
        texts = self.decoder.decode(probs, method='greedy')

        # Compute confidences
        if return_confidence:
            confidences = self._compute_confidence(probs)
        else:
            confidences = [1.0] * len(texts)

        # Parse results
        results = []
        for text, conf in zip(texts, confidences):
            # Convert text to integer
            number = self._text_to_number(text)

            results.append({
                'number': number,
                'text': text,
                'confidence': conf
            })

        return results

    def _compute_confidence(self, probs: torch.Tensor) -> List[float]:
        """
        Compute confidence scores from probabilities.

        Args:
            probs: Log probabilities (T, B, num_classes)

        Returns:
            Confidence scores for each sample
        """
        # Take max probability at each timestep and average
        max_probs, _ = probs.max(dim=2)  # (T, B)
        max_probs = torch.exp(max_probs)  # Convert from log
        confidences = max_probs.mean(dim=0)  # Average over time

        return confidences.cpu().tolist()

    def _text_to_number(self, text: str) -> Optional[int]:
        """
        Convert recognized text to jersey number.

        Args:
            text: Recognized text (e.g., '7', '10', '99')

        Returns:
            Jersey number as integer, or None if invalid
        """
        if not text or not text.isdigit():
            return None

        number = int(text)

        # Valid jersey numbers: 1-99
        if 1 <= number <= 99:
            return number
        else:
            return None

    def recognize_single(self, image: np.ndarray) -> dict:
        """
        Recognize jersey number from a single image.

        Args:
            image: Number region crop (H, W, 3)

        Returns:
            Dict with number, text, and confidence
        """
        results = self.recognize(image[np.newaxis])
        return results[0]


class TemporalNumberAggregator:
    """Aggregates jersey number predictions over time for robustness.

    Jersey numbers are only visible ~30% of frames, so we need to
    aggregate predictions over time.
    """

    def __init__(self, window_size: int = 30, min_confidence: float = 0.5):
        """
        Args:
            window_size: Number of frames to aggregate over
            min_confidence: Minimum confidence to consider a prediction
        """
        self.window_size = window_size
        self.min_confidence = min_confidence

        # Track history per track_id
        self.history = {}  # track_id -> List[(number, confidence)]

    def add_prediction(self, track_id: int, number: Optional[int],
                      confidence: float):
        """
        Add a prediction for a track.

        Args:
            track_id: Track ID
            number: Predicted number (or None)
            confidence: Prediction confidence
        """
        if track_id not in self.history:
            self.history[track_id] = []

        # Only add high-confidence predictions with valid numbers
        if number is not None and confidence >= self.min_confidence:
            self.history[track_id].append((number, confidence))

            # Limit history size
            if len(self.history[track_id]) > self.window_size:
                self.history[track_id].pop(0)

    def get_stable_number(self, track_id: int,
                         min_votes: int = 3) -> Optional[int]:
        """
        Get stable jersey number for a track using voting.

        Args:
            track_id: Track ID
            min_votes: Minimum number of votes required

        Returns:
            Most common jersey number, or None if not enough votes
        """
        if track_id not in self.history or len(self.history[track_id]) < min_votes:
            return None

        # Weight votes by confidence
        votes = {}
        for number, conf in self.history[track_id]:
            if number not in votes:
                votes[number] = 0.0
            votes[number] += conf

        # Get most voted number
        best_number = max(votes.items(), key=lambda x: x[1])[0]

        # Check if we have enough votes
        num_votes = sum(1 for n, _ in self.history[track_id] if n == best_number)
        if num_votes >= min_votes:
            return best_number
        else:
            return None

    def get_confidence(self, track_id: int, number: int) -> float:
        """
        Get confidence for a specific number assignment.

        Args:
            track_id: Track ID
            number: Jersey number

        Returns:
            Confidence score [0, 1]
        """
        if track_id not in self.history or len(self.history[track_id]) == 0:
            return 0.0

        # Average confidence of predictions for this number
        confidences = [conf for n, conf in self.history[track_id] if n == number]

        if len(confidences) == 0:
            return 0.0

        return sum(confidences) / len(confidences)

    def clear_track(self, track_id: int):
        """Clear history for a track."""
        if track_id in self.history:
            del self.history[track_id]
