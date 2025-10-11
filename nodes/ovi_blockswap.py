"""ComfyUI helper node for BlockSwap settings."""

BLOCK_COUNT = 32


class OviBlockSwap:
    CATEGORY = "Ovi"
    RETURN_TYPES = ("OVI_BLOCKSWAP",)
    RETURN_NAMES = ("bswap",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_blockswap": ("BOOLEAN", {"default": True}),
                "blocks_per_chunk": ("INT", {"default": 1, "min": 1, "max": BLOCK_COUNT, "display": "slider"}),
            },
        }

    def build(self, enable_blockswap: bool, blocks_per_chunk: int):
        chunk = max(1, min(int(blocks_per_chunk), BLOCK_COUNT))
        if chunk != blocks_per_chunk:
            print(
                f"[OVI] Adjusting blocks_per_chunk from {blocks_per_chunk} to {chunk} (total blocks: {BLOCK_COUNT})."
            )
        return ({
            "enabled": bool(enable_blockswap),
            "blocks_per_chunk": chunk,
            "total_blocks": BLOCK_COUNT,
        },)
