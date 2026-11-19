"""Device mask construction for the eight-rank S1 qualification."""

def build_ffn_mask(torch_module, full_rows, selected_ids, *, all_selected=False):
    """Build the bool mask from caller-owned device indices without index_put.

    The caller supplies sorted, valid row IDs; all_selected means those IDs are
    exactly arange(full_rows). Keep scalar values in device fill kernels rather than
    Python advanced-index assignment, whose CUDA scalar path may synchronize.
    This helper alone does not certify GPU execution or validate a row plan.
    """
    t = torch_module
    if all_selected:
        return t.ones(full_rows, device=selected_ids.device, dtype=t.bool)
    return t.zeros(full_rows, device=selected_ids.device, dtype=t.bool).index_fill_(0, selected_ids, True)
