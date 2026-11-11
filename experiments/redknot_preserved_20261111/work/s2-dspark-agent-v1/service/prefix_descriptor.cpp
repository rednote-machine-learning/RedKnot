// Read-only host bridge to pinned native structures. No allocation/scheduler
// mutation, CUDA operation, kernel, or reinterpret_cast is exposed.
#include <pybind11/stl.h>
#include "engine/sequence.hpp"

py::dict sequence_descriptor(const Sequence &seq) {
    py::dict d;
    d["request_id"] = seq.request.request_id;
    d["block_table"] = seq.cache.block_table;
    d["num_cached_tokens"] = seq.cache.num_tokens;
    d["previous_num_tokens"] = seq.cache.previous_num_tokens;
    d["state_id"] = seq.cache.state_id;
    d["swa_valid_start"] = seq.cache.swa_valid_start;
    d["block_size"] = seq.cache.block_size;
    d["last_copy_state_token_idx"] = seq.cache.last_copy_state_token_idx;
    d["prompt_tokens"] = seq.get_prompt_len();
    d["output_tokens"] = seq.get_output_len();
    d["schedule_status"] = seq.schedule.status == SequenceStatus::WAITING ? "WAITING" :
        (seq.schedule.status == SequenceStatus::RUNNING ? "RUNNING" : "OTHER");
    d["total_inflight_tokens"] = seq.schedule.total_inflight_tokens;
    d["last_inflight_tokens"] = seq.schedule.last_inflight_tokens;
    d["mtp_info_slot"] = seq.mtp.info_slot;
    d["mtp_has_draft_info"] = seq.mtp.has_draft_info;
    d["mtp_last_output_token"] = seq.mtp.last_output_token;
    d["mtp_accepted_count"] = seq.mtp.accepted_tokens.size();
    d["mtp_draft_count"] = seq.mtp.draft_tokens.size();
    d["mtp_accept_rate_count"] = seq.mtp.accept_rates.size();
    d["mtp_prev_accept_rate_count"] = seq.mtp.prev_accept_rates.size();
    d["mtp_confidence_count"] = seq.mtp.confidence_raw.size();
    d["mtp_prev_confidence_count"] = seq.mtp.prev_confidence_raw.size();
    d["mtp_process_count"] = seq.mtp.process_cnt;
    d["mtp_schedule_count"] = seq.mtp.schedule_cnt;
    return d;
}

py::dict metadata_descriptor(const SequenceMetadata &meta) {
    py::dict d;
    d["request_id"] = meta.request_id;
    d["block_table"] = meta.block_table;
    d["state_id"] = meta.state_id;
    d["swa_valid_start"] = meta.swa_valid_start;
    d["is_prompt"] = meta.is_prompt;
    d["is_placeholder"] = meta.is_placeholder;
    d["seqlen_q"] = meta.seqlen_q;
    d["seqlen_k"] = meta.seqlen_k;
    d["mtp_info_read_slot"] = meta.mtp_info_read_slot;
    d["mtp_info_write_slot"] = meta.mtp_info_write_slot;
    return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sequence_descriptor", &sequence_descriptor);
    m.def("metadata_descriptor", &metadata_descriptor);
    m.attr("read_only") = true;
    m.attr("descriptor_contract") = "fresh-sequence-dspark-prefix-v1";
    m.attr("sequence_sizeof") = py::int_(sizeof(Sequence));
    m.attr("metadata_sizeof") = py::int_(sizeof(SequenceMetadata));
}
