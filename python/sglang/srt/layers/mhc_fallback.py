import torch


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    b, s, _ = mixes.size()
    flat = mixes.view(-1, (2 + hc_mult) * hc_mult).float()

    pre = torch.sigmoid(flat[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post_start = hc_mult
    post_end = 2 * hc_mult
    post = 2 * torch.sigmoid(
        flat[:, post_start:post_end] * hc_scale[1] + hc_base[post_start:post_end]
    )

    comb_start = 2 * hc_mult
    comb_logits = flat[:, comb_start:].view(-1, hc_mult, hc_mult) * hc_scale[
        2
    ] + hc_base[comb_start:].view(hc_mult, hc_mult)
    comb = torch.softmax(comb_logits, dim=2) + eps
    col_sum = comb.sum(dim=1, keepdim=True)
    comb = comb / (col_sum + eps)

    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(dim=2, keepdim=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(dim=1, keepdim=True)
        comb = comb / (col_sum + eps)

    return (
        pre.view(b, s, hc_mult).to(mixes.dtype),
        post.view(b, s, hc_mult).to(mixes.dtype),
        comb.view(b, s, hc_mult, hc_mult).to(mixes.dtype),
    )
