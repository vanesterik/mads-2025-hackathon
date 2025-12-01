import json
import os

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from loguru import logger
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer


class NebiusTextVectorizer:
    """
    Wraps the Nebius Token Factory / OpenAI compatible API for text vectorization.
    """

    def __init__(
        self,
        model_name: str,
        # max_length and pooling are ignored as they are handled by the API
        max_length: int | None = None,
        pooling: str | None = None,
    ):
        """
        Initializes the client and specifies the model to use.
        The max_length and pooling arguments are kept for compatibility but ignored.
        """
        self.model_name = model_name

        # 1. Initialize the OpenAI client pointing to the Nebius endpoint
        self.client = OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=os.environ.get("NEBIUS_API_KEY"),
        )

        # 2. Log initialization details
        # Note: We can't determine the hidden_size easily without an extra API call
        # or checking the model's documentation. We default to a common size.
        # A quick check can be done by calling self.hidden_size once.
        self._hidden_size = None

        logger.info(
            f"TextVectorizer initialized for Nebius model={self.model_name}. "
            "Max length and pooling are managed by the API."
        )

    # The _detect_pooling_strategy and all tokenization/forward logic are removed.

    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Encodes a list of texts into vectors using the Nebius Embeddings API.
        """
        if not texts:
            return torch.empty((0, self.hidden_size))

        # 1. API Call to get embeddings
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts,  # The API accepts a list of strings directly
            )
        except Exception as e:
            logger.error(f"Error calling Nebius API for embeddings: {e}")
            raise e

        # 2. Extract and format the embeddings
        # The response structure is: response.data[i].embedding (a list of floats)

        # Get the list of embeddings (list of lists of floats)
        embeddings_list = [item.embedding for item in response.data]

        # Convert the list of lists into a PyTorch tensor
        vector_tensor = torch.tensor(embeddings_list, dtype=torch.float32)

        # 3. Cache the hidden size after the first successful call
        if self._hidden_size is None and vector_tensor.shape[1] > 0:
            self._hidden_size = vector_tensor.shape[1]
            logger.info(f"Auto-detected hidden_size: {self._hidden_size}")

        return vector_tensor

    @property
    def hidden_size(self) -> int:
        """
        Returns the embedding dimension. Makes an API call on first access if not known.
        """
        if self._hidden_size is not None:
            return self._hidden_size

        # If not known, perform a test API call with a dummy input
        logger.info("Determining hidden_size with a test API call...")
        dummy_input = ["test sentence"]

        try:
            vector = self.forward(dummy_input)
            self._hidden_size = vector.shape[1]
            return self._hidden_size
        except Exception as e:
            logger.error(f"Could not determine hidden_size via API call: {e}")
            # Fallback to a common size if API fails, though this is risky
            return 768  # Common hidden size for models like BERT/BGE


class TextVectorizer:
    """
    Wraps a HuggingFace model to vectorize text.
    """

    def __init__(
        self, model_name: str, max_length: int | None = None, pooling: str | None = None
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set to eval mode by default

        if pooling is None:
            self.pooling = self._detect_pooling_strategy(model_name)
        else:
            self.pooling = pooling

        # Determine max_length
        if max_length is not None:
            self.max_length = max_length
        else:
            # Try to get from tokenizer
            model_max_len = self.tokenizer.model_max_length
            # HuggingFace tokenizers often return a very large int if not set
            # We treat anything > 10000 as "infinite"/unset and fallback to 512
            if model_max_len > 10000:
                logger.warning(
                    f"Tokenizer model_max_length is {model_max_len}, falling back to 512. "
                    "Specify max_length explicitly if needed."
                )
                self.max_length = 512
            else:
                self.max_length = model_max_len

        logger.info(
            f"TextVectorizer initialized with max_length={self.max_length}, pooling={self.pooling}"
        )

    def _detect_pooling_strategy(self, model_name: str) -> str:
        """
        Attempts to detect the pooling strategy from the model's configuration.
        """
        try:
            # Download modules.json
            path = hf_hub_download(model_name, "modules.json")
            with open(path) as f:
                modules = json.load(f)

            # Look for the pooling module
            for module in modules:
                if module["type"] == "sentence_transformers.models.Pooling":
                    # Download the pooling config
                    config_path = hf_hub_download(
                        model_name, f"{module['path']}/config.json"
                    )
                    with open(config_path) as f:
                        config = json.load(f)

                    if config.get("pooling_mode_cls_token"):
                        logger.success(
                            f"Auto-detected pooling strategy: cls (from {model_name})"
                        )
                        return "cls"
                    if config.get("pooling_mode_mean_tokens"):
                        logger.success(
                            f"Auto-detected pooling strategy: mean (from {model_name})"
                        )
                        return "mean"
        except Exception:
            # Fail silently on network errors or missing files
            pass

        logger.warning(
            f"Could not detect pooling strategy for {model_name}, defaulting to 'mean'. "
            "Please check the model card to see if 'cls' pooling is required."
        )
        return "mean"

    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Encodes a list of texts into vectors.
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        if self.pooling == "cls":
            # CLS token is usually the first token
            return outputs.last_hidden_state[:, 0, :]

        # Default to mean pooling
        # batches of text are padded such that they have the same length,
        # however, the padding tokens are zero and we dont want to include them in the mean
        attention_mask = inputs["attention_mask"]
        # attention_mask shape: (batch_size, seq_len)
        token_embeddings = outputs.last_hidden_state
        # last_hidden_state shape: (batch_size, seq_len, hidden_dim)

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        # Zero out padding tokens so they don't skew the sum, then
        # divide by the count of real tokens (not total length) for a true average.
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size


class NeuralClassifier(nn.Module):
    """
    A simple neural network for classification.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Returns logits.
        """
        x = self.sequential(x)
        return x


class ResidualBlock(nn.Module):
    """
    A helper block that implements the Skip Connection and Batch Normalization.
    """

    def __init__(self, hidden_dim, dropout=0.1):  # FIXED: _init_ -> __init__
        super().__init__()  # FIXED: super()._init_() -> super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Batch Norm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Batch Norm before addition
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x  # 1. Save input (Skip connection)
        out = self.block(x)  # 2. Process
        out += identity  # 3. Add input to output
        out = self.relu(out)  # 4. Final Activation
        return out


class HybridClassifier(nn.Module):
    """
    A classifier that combines text embeddings and regex features
    using Residual Connections and Batch Normalization.
    """

    def __init__(
        self, input_dim: int, regex_dim: int, num_classes: int, hidden_dim: int = 512
    ):
        super().__init__()

        # Concatenated input dimension
        combined_dim = input_dim + regex_dim
        logger.info(f"Combined dimension: {combined_dim}")

        # 1. Input Projection
        # Transforms the 'combined_dim' (e.g., 800) to 'hidden_dim' (e.g., 128).
        # We cannot use a skip connection here because sizes differ.
        self.input_projection = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )

        # 2. Residual Layer (Skip Layer)
        # Input is 128, Output is 128. This allows us to add the skip connection.
        self.residual_layer = ResidualBlock(hidden_dim, dropout=0.1)

        # 3. Output Head
        # Transforms 'hidden_dim' to 'num_classes'
        self.output_head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self, text_emb: torch.Tensor, regex_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass. Concatenates inputs and returns logits.
        """
        # 1. Concatenate along the feature dimension (dim=1)
        combined = torch.cat((text_emb, regex_feats), dim=1)

        # 2. Project to hidden dimension
        x = self.input_projection(combined)

        # 3. Apply Residual Block (with skip connection)
        x = self.residual_layer(x)

        # 4. Final Classification
        logits = self.output_head(x)

        return logits
