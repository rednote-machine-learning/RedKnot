"""Natural tool-document admission, without changing frozen S1 mathematics."""
from __future__ import annotations

from kv_cache_engram.core import digest, require, token_digest, validate_digest
from kv_cache_engram.s1_plan import all_chunks

MIN_DOCUMENT_TOKENS = 256
GROUP_CHUNKS = 17
MAX_SELECTED_ROWS = 8192
MAX_BANK_BYTES = 4 * 1024**3
SCOPE_KEYS = ('model_sha256', 'execution_sha256', 'tokenizer_sha256')


def validate_scope(scope):
    require(type(scope) is dict and set(scope) == set(SCOPE_KEYS), 'exact model/execution/tokenizer scope required')
    for key in SCOPE_KEYS:
        validate_digest(scope[key], key)
    return dict(scope)


def natural_documents(input_ids, documents):
    """Validate token occurrences; never derive spans by substring guessing.

    Repeated token content has one immutable bank key and multiple positions.
    Short documents remain unchanged in the prompt and are not cacheable.
    """
    require(isinstance(input_ids, (list, tuple)) and input_ids
            and all(type(v) is int and 0 <= v < 2**31 for v in input_ids), 'nonempty int32 token input required')
    require(isinstance(documents, (list, tuple)), 'explicit document occurrence list required')
    previous = 0
    out = []
    for ordinal, doc in enumerate(documents):
        require(type(doc) is dict, 'document occurrence must be a mapping')
        start, end = doc.get('input_start'), doc.get('input_end')
        require(type(start) is int and type(end) is int and previous <= start < end <= len(input_ids),
                'document occurrences must be ordered, disjoint and in bounds')
        key = token_digest(input_ids[start:end])
        require(doc.get('token_ids_sha256') == key and doc.get('document_id', key) == key
                and doc.get('offline_position', 0) == 0 and doc.get('online_position', start) == start
                and doc.get('tokens', end-start) == end-start, 'document token or position identity differs')
        out.append(dict(occurrence=ordinal, input_start=start, input_end=end, tokens=end-start,
                        document_id=key, token_ids_sha256=key, offline_position=0, online_position=start))
        previous = end
    return out


def s2_documents(input_ids, documents, *, expected_document_count=None):
    docs = natural_documents(input_ids, documents or ())
    require(expected_document_count is None or expected_document_count == len(docs), 'request document count differs')
    require(all(d['tokens'] >= MIN_DOCUMENT_TOKENS for d in docs), 'short documents must remain native nondocument rows')
    return tuple(dict(start=d['input_start'], end=d['input_end'], key=d['document_id']) for d in docs)


def uniform_bank_budget(free_bytes_by_rank):
    require(type(free_bytes_by_rank) in (list, tuple) and len(free_bytes_by_rank) == 8
            and all(type(n) is int and n >= 0 for n in free_bytes_by_rank), 'all eight actual free-memory readings required')
    return min(MAX_BANK_BYTES, min(free_bytes_by_rank)//5)


def plan_request(input_ids, documents, scope, *, bank_budget_bytes=None,
                 resident_bank_bytes=0, resident_document_ids=(), session_certificate=None):
    scope = validate_scope(scope)
    require(bank_budget_bytes is None or type(bank_budget_bytes) is int and 0 <= bank_budget_bytes <= MAX_BANK_BYTES,
            'bank budget must be actual uniform cap no greater than4GiB')
    require(type(resident_bank_bytes) is int and resident_bank_bytes >= 0, 'nonnegative actual resident payload bytes required')
    require(type(resident_document_ids) in (tuple, list) and len(set(resident_document_ids)) == len(resident_document_ids),
            'unique complete resident document identities required')
    for key in resident_document_ids: validate_digest(key, 'resident document identity')
    docs = natural_documents(input_ids, documents)
    eligible = [d for d in docs if d['tokens'] >= MIN_DOCUMENT_TOKENS]
    short = [d for d in docs if d['tokens'] < MIN_DOCUMENT_TOKENS]
    chunks = all_chunks(s2_documents(input_ids, eligible), len(input_ids)) if eligible else []
    group_rows = [sum(c['selected_rows'] for c in chunks[i:i+GROUP_CHUNKS])
                  for i in range(0, len(chunks), GROUP_CHUNKS)]
    unique = {}
    for d in eligible: unique.setdefault(d['document_id'], d)
    missing = [key for key in unique if key not in set(resident_document_ids)]
    predicted_miss = sum(37*128*512*2 + 9216*unique[key]['tokens'] for key in missing)
    session_sha = domain = domain_reason = None
    if session_certificate is not None:
        session_sha = session_certificate.get('sha256')
        validate_digest(session_sha, 'session certificate')
        domain = session_certificate.get('supported_geometry')
        keys = ('max_input_tokens', 'max_document_occurrences', 'max_document_tokens', 'max_group_selected_rows')
        require(type(domain) is dict and set(domain) == set(keys)
                and all(type(n) is int and n > 0 for n in domain.values()), 'explicit session geometry bounds required')
        observed = (len(input_ids), len(eligible), max((d['tokens'] for d in eligible), default=0), max(group_rows, default=0))
        domain_reason = next(('session_'+key+'_exceeded' for key, actual in zip(keys, observed) if actual > domain[key]), None)
    reason = ('no_cacheable_document' if not eligible else
              'group17_selected_rows_exceed_8192' if any(n > MAX_SELECTED_ROWS for n in group_rows) else
              'bank_budget_unavailable' if bank_budget_bytes is None else
              'bank_budget_exceeded' if resident_bank_bytes + predicted_miss > bank_budget_bytes else domain_reason)
    route = 'full_fallback' if reason else 's2'
    effective = eligible if route == 's2' else []
    admitted_unique = dict((d['document_id'], d) for d in effective)
    body = dict(schema='s2-natural-request-plan-v1', requested_arm='s2', route=route,
                effective_arm='full' if reason else 's2', fallback_reason=reason,
                input_ids_sha256=token_digest(input_ids), input_tokens=len(input_ids), scope=scope,
                source_documents=docs, effective_documents=effective, short_documents=short,
                unique_document_ids=list(admitted_unique), group_chunks=GROUP_CHUNKS, group_selected_rows=group_rows,
                native_selected_row_capacity=MAX_SELECTED_ROWS, padding_tokens_added=0,
                document_contents_rewritten=False, short_documents_recomputed_as_nondocument=True,
                unique_bank_payload_bytes=sum(37*128*512*2 + 9216*d['tokens'] for d in unique.values()),
                bank_budget_bytes=bank_budget_bytes, resident_bank_bytes=resident_bank_bytes,
                resident_document_ids=sorted(resident_document_ids), predicted_missing_document_ids=missing,
                predicted_miss_payload_bytes=predicted_miss,
                predicted_retained_payload_bytes=resident_bank_bytes + predicted_miss,
                memory_admission_complete=bank_budget_bytes is not None, eviction_performed=False,
                session_certificate_sha256=session_sha, session_supported_geometry=domain,
                numerical_qualification_claim=False)
    return dict(body, sha256=digest(body))


def validate_plan(plan, *, scope=None, input_ids=None):
    require(type(plan) is dict and plan.get('schema') == 's2-natural-request-plan-v1'
            and plan.get('sha256') == digest({k: v for k, v in plan.items() if k != 'sha256'}), 'invalid S2 plan seal')
    validate_scope(plan['scope'])
    if scope is not None:
        require(plan['scope'] == scope, 'plan belongs to another model/execution/tokenizer')
    if input_ids is not None:
        require(plan == plan_request(input_ids, plan['source_documents'], plan['scope'],
                    bank_budget_bytes=plan['bank_budget_bytes'], resident_bank_bytes=plan['resident_bank_bytes'],
                    resident_document_ids=plan['resident_document_ids'], session_certificate=None if plan['session_certificate_sha256'] is None
                        else dict(sha256=plan['session_certificate_sha256'], supported_geometry=plan['session_supported_geometry'])),
                'request token identity or geometry changed')
    return plan
