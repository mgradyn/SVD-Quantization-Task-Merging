from dataclasses import dataclass

@dataclass
class SVDHybridConfig:
    energy_threshold: float = 0.90        # retain ≥ this fraction of cumulative energy
    max_rank: int = 32                    # cap rank
    fp_dtype: str = "fp16"                # storage dtype for bases & c_high
    center_task_matrix: bool = True       # mean-center task columns
    store_singular_values: bool = True
    selected_blocks: tuple = (0, 6, 11)   # subset of transformer blocks for light run
    layer_types: tuple = ("attn.in_proj_weight", "mlp.fc1.weight", "mlp.fc2.weight")
    quant_bits_low: int = 2               # RTVQ bits for low-energy coefficients
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"