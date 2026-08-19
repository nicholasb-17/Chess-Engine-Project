#Resnet CNN
# PyTorch policy/value network for the chess engine.
# Input: 106 x 8 x 8 tensor produced by dataset.py's board_to_tensor().
# Output: policy logits over a fixed move-encoding space, and a scalar value estimate.
#
# When writing the training loop (separate from this file), wrap the
# forward pass in amp_context(device) too, and pair fp16 CUDA autocast
# with torch.cuda.amp.GradScaler on the backward/optimizer step -- that's
# where Colab's T4 Tensor Cores give the biggest speedup. infer()/
# infer_batch() below already do this for MCTS-time evaluation.
import contextlib

import torch
import torch.nn as nn #for classes
import torch.nn.functional as F #for functions

from engine_output_dataset import POLICY_OUTPUT_SIZE    

# Global Variables for the network architecture
INPUT_CHANNELS = 106     # must match dataset.py's board_to_tensor() output channel count
NUM_FILTERS = 192        # width of the residual tower
NUM_RES_BLOCKS = 16      # depth of the residual tower
SE_RATIO = 8             # squeeze-and-excitation channel reduction ratio inside each
                         # residual block (channels -> channels // SE_RATIO -> channels)
POLICY_CHANNELS = 32     # channels in the policy head's conv reduction
VALUE_CHANNELS = 4       # channels in the value head's conv reduction
VALUE_HIDDEN = 256       # hidden units in the value head's fully-connected layer


def get_device() -> torch.device:
    """
    Returns device("cuda") if a CUDA GPU is available, ("xpu") if an XPU is available, or device("cpu") otherwise.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def amp_context(device: torch.device):
    """
    Returns an appropriate context manager for Automatic Mixed Precision (AMP).

    Selects the optimal floating-point format for PyTorch autocasting based on
    the target hardware device, falling back to a no-op context manager when
    autocasting is unsupported or unbeneficial.
    """
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if device.type == "xpu":
        return torch.autocast(device_type="xpu", dtype=torch.float16)
    if device.type == "cpu":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


class SqueezeExcitation(nn.Module):
    """
    Performs feature recalibration by globally pooling spatial dimensions to 
    summarize channel-wise statistics ("squeeze"), passing the resulting vector 
    through a two-layer bottleneck MLP, and applying a sigmoid gate ("excitation") 
    to dynamically weight individual channels.
    """ 

    def __init__(self, channels: int, ratio: int = SE_RATIO):
        super().__init__()
        #Channel reduction staying above 0
        reduced = max(channels // ratio, 1)
        #bottleneck MLP: channels -> reduced
        self.fc_reduce = nn.Linear(channels, reduced)
        #maps back to original channel count, producing a sigmoid gate for each channel
        self.fc_expand = nn.Linear(reduced, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        #squeeze
        pooled = x.mean(dim=(2, 3))  # (batch, channels) -- global average pool per channel on height/width
        #excitation
        gate = F.relu(self.fc_reduce(pooled))
        gate = torch.sigmoid(self.fc_expand(gate)).view(b, c, 1, 1)
        #rescale each channel of the original feature map by its learned importance score
        return x * gate


class ResidualBlock(nn.Module):
    """
    Residual block with an integrated Squeeze-and-Excitation (SE) channel attention gate.

    Computes a residual transformation with the following sequential pipeline:
    `Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> SE -> (+ Identity) -> ReLU`.
    """

    def __init__(self, channels: int, se_ratio: int = SE_RATIO):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        nn.init.zeros_(self.bn2.weight)  # zero-init the residual branch
        self.se = SqueezeExcitation(channels, ratio=se_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + identity  # the residual (skip) connection using element wise addition
        return F.relu(out)  


class PolicyHead(nn.Module):
    """
    Reduces the trunk down to a small number of channels, then flattens
    and projects to POLICY_OUTPUT_SIZE logits (one per possible move
    encoding). Illegal moves should be masked out at inference time by
    the caller, not by this head.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        #channel reduction conv layer, followed by batch normalization and ReLU activation
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=POLICY_CHANNELS,
                              kernel_size=1,
                              bias=False)
        self.bn = nn.BatchNorm2d(POLICY_CHANNELS)
        #fully connected linear layer
        self.fc = nn.Linear(in_features=POLICY_CHANNELS * 8 * 8,
                            out_features=POLICY_OUTPUT_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn(self.conv(x)))
        out = out.flatten(start_dim=1)
        return self.fc(out)  # raw logits; apply softmax/cross-entropy outside


class ValueHead(nn.Module):
    """
    Reduces the trunk down to a small number of channels, flattens,
    passes through a small MLP, and squashes to a scalar in [-1, 1] to represent the evaluation of the position
    via tanh (from the perspective of the side to move.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        #channel reduction conv layer, followed by batch normalization and ReLU activation
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=VALUE_CHANNELS,
                              kernel_size=1,
                              bias=False)
        self.bn = nn.BatchNorm2d(VALUE_CHANNELS)
        self.fc1 = nn.Linear(in_features=VALUE_CHANNELS * 8 * 8,
                             out_features=VALUE_HIDDEN)
        self.fc2 = nn.Linear(in_features=VALUE_HIDDEN,
                             out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn(self.conv(x)))
        out = out.flatten(start_dim=1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return torch.tanh(out).squeeze(-1)  # shape (batch)


class ChessNet(nn.Module):
    """
    Full policy/value network.

    Stem: conv3x3 (INPUT_CHANNELS -> NUM_FILTERS) + BN + ReLU
    Trunk: NUM_RES_BLOCKS basic residual blocks
    Heads: PolicyHead and ValueHead, both branching off the trunk output

    forward() returns (policy_logits, value) so a single forward pass
    produces both training targets' predictions.
    """

    def __init__(self,
        input_channels: int = INPUT_CHANNELS,
        num_filters: int = NUM_FILTERS,
        num_res_blocks: int = NUM_RES_BLOCKS,
        se_ratio: int = SE_RATIO,
        device: str | torch.device | None = None,):

        super().__init__()
        self.stem_conv = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(num_filters)
        self.res_blocks = nn.ModuleList([ResidualBlock(num_filters, se_ratio=se_ratio) for _ in range(num_res_blocks)])
        self.policy_head = PolicyHead(num_filters)
        self.value_head = ValueHead(num_filters)

        # Auto-detect the best device (Intel XPU / CUDA / CPU) unless the
        # caller specifies one explicitly.
        self.to(device if device is not None else get_device())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = F.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.res_blocks:
            out = block(out)
        policy_logits = self.policy_head(out)
        value = self.value_head(out)
        return policy_logits, value

    @staticmethod
    def predict_policy(policy_logits: torch.Tensor, legal_move_indices: list[list[int]]) -> torch.Tensor:
        """
        Masks illegal moves out of the raw policy logits and softmaxes the
        remainder, per position in the batch.

        policy_logits: (batch, POLICY_OUTPUT_SIZE) raw logits from forward().
        legal_move_indices[i]: list of policy indices that are legal for
            batch item i (produced by encode_move() in
            engine_output_dataset.py over board.legal_moves).

        Returns a (batch, POLICY_OUTPUT_SIZE) tensor of probabilities that
        sum to 1 over each position's legal moves and are exactly 0 elsewhere.

        Raises ValueError if any entry in legal_move_indices is empty --
        this means the caller is asking for a policy on a position with no
        legal moves (checkmate/stalemate). Softmax over an all -inf row is
        NaN by construction, and returning NaN silently is far worse than
        failing loudly: callers (MCTS/search code) must check
        board.is_game_over() before calling infer()/infer_batch(), and a
        terminal leaf should never reach the policy network at all.
        """
        if any(len(indices) == 0 for indices in legal_move_indices):
            raise ValueError(
                "predict_policy() received a position with zero legal moves "
                "(checkmate/stalemate). Terminal positions must be handled "
                "by the caller (e.g. MCTS) before reaching the network -- "
                "softmax over an all -inf row is undefined (NaN)."
            )
        # Build the mask in float32 regardless of policy_logits' dtype so a
        # reduced-precision (fp16/bf16) autocast context upstream can't
        # silently downgrade the softmax's precision or the returned dtype.
        mask = torch.full_like(policy_logits, float("-inf"), dtype=torch.float32)
        for i, indices in enumerate(legal_move_indices):
            mask[i, indices] = 0.0
        return F.softmax(policy_logits.float() + mask, dim=-1)

    @torch.inference_mode() #turn off gradient tracking for inference, saving memory and speeding up evaluation
    def infer(self, tensor: torch.Tensor, legal_move_indices: list[int]) -> tuple[torch.Tensor, float]:
        """
        Single-position convenience inference: takes one (INPUT_CHANNELS, 8, 8)
        board tensor and its legal move indices, returns (policy probs over
        POLICY_OUTPUT_SIZE, scalar value). Puts the model in eval mode and
        disables gradient tracking. Intended for search code that needs to
        evaluate one leaf at a time.
        """
        self.eval()
        device = next(self.parameters()).device
        x = tensor.unsqueeze(0).to(device)
        with amp_context(device):
            policy_logits, value = self.forward(x)
        policy = self.predict_policy(policy_logits, [legal_move_indices])
        return policy.squeeze(0), value.item()

    @torch.inference_mode()
    def infer_batch(
        self, tensors: torch.Tensor, legal_move_indices: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Batched inference for MCTS leaf evaluation: takes a stacked batch of
        board tensors (batch, INPUT_CHANNELS, 8, 8) and each position's legal
        move indices, returns (policy probs, values) for the whole batch in
        one forward pass. Search code should accumulate several leaves (e.g.
        via virtual loss) and call this once rather than calling infer()
        per-leaf, since a single batched GPU call is far cheaper than many
        small ones.
        """
        self.eval()
        device = next(self.parameters()).device
        x = tensors.to(device)
        with amp_context(device):
            policy_logits, value = self.forward(x)
        policy = self.predict_policy(policy_logits, legal_move_indices)
        return policy, value

    def save_checkpoint(self, path: str, **extra) -> None:
        """
        Saves model weights plus the architecture hyperparameters needed to
        reconstruct this exact ChessNet, plus any extra metadata the caller
        wants recorded (e.g. optimizer state, epoch, training step).
        """
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "input_channels": INPUT_CHANNELS,
                "num_filters": NUM_FILTERS,
                "num_res_blocks": NUM_RES_BLOCKS,
                "se_ratio": SE_RATIO,
                **extra,
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str, device: str | torch.device | None = None) -> "ChessNet":
        """
        Loads a checkpoint saved by save_checkpoint(), reconstructs the
        ChessNet with matching architecture hyperparameters, and moves it
        to the requested device in eval mode. If device is not specified,
        auto-detects the best available device (Intel XPU / CUDA / CPU).
        """
        device = device if device is not None else get_device()
        checkpoint = torch.load(path, map_location=device)
        net = cls(
            input_channels=checkpoint.get("input_channels", INPUT_CHANNELS),
            num_filters=checkpoint.get("num_filters", NUM_FILTERS),
            num_res_blocks=checkpoint.get("num_res_blocks", NUM_RES_BLOCKS),
            se_ratio=checkpoint.get("se_ratio", SE_RATIO),
            device=device,
        )
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        return net


# sanity checks
if __name__ == "__main__":
    torch.manual_seed(0)
    net = ChessNet()
    device = next(net.parameters()).device
    print(f"using device: {device}")
    num_params = sum(p.numel() for p in net.parameters())
    print(f"total parameters: {num_params:,}")

    # fake batch of 4 positions, matching dataset.py's (106, 8, 8) tensor shape
    dummy_input = torch.randn(4, INPUT_CHANNELS, 8, 8).to(device)
    policy_logits, value = net(dummy_input)
    print("policy_logits shape:", policy_logits.shape)  # expect (4, 4672)
    assert policy_logits.shape == (4, POLICY_OUTPUT_SIZE)
    print("value shape:", value.shape)  # expect (4)
    assert value.shape == (4,)
    print("value range check (should be within [-1, 1]):", value.min().item(), value.max().item())
    assert (value.abs() <= 1.0).all()

    # zero-init check: with bn2.weight zeroed, each residual block should be the identity function at initialization, before any training happens
    block = ResidualBlock(NUM_FILTERS)
    block.eval()  # use running stats instead of batch stats for a clean identity check
    probe = torch.randn(2, NUM_FILTERS, 8, 8)
    with torch.inference_mode():
        block_out = block.bn1(block.conv1(probe))  # unused, just exercising the path
    with torch.inference_mode():
        out = probe
        identity = out
        out = F.relu(block.bn1(block.conv1(out)))
        out = block.bn2(block.conv2(out))
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5), "bn2 should zero out the residual branch"
        result = F.relu(out + identity)
        assert torch.allclose(result, F.relu(identity)), "block should act as identity (post-ReLU) at init"
    print("zero-init residual check: each block behaves as identity at initialization, as expected")
    print(hasattr(torch, "xpu"), torch.xpu.is_available() if hasattr(torch, "xpu") else None)