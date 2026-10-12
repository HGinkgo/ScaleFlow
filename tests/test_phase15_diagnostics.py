from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import pytest

from scaleflow.phase15_diagnostics import (
    METHODS, aggregate, build_prompt, chat_token_ids, exclusive_json,
    extract_final, output_flags, phase15_contexts, validate_budget,
)
from scaleflow.phase15_worker import generate_one
from scaleflow.rag import score_answer


@pytest.mark.parametrize(('raw', 'expected'), [
    ('FINAL: 42', '42'), ('FINAL: unknown\n', 'unknown'),
    ("thinking\nFINAL: King's College", "King's College"),
    ('42', None), ('FINAL: \n42', None), ('FINAL:', None),
    ('not FINAL: 42', None), ('FINAL: first\nFINAL: last', 'last'),
])
def test_final_line_extraction(raw, expected):
    assert extract_final(raw) == expected


@pytest.mark.parametrize(('tokens', 'finish', 'reached'), [
    (255, 'stop', False), (256, 'length', True), (257, None, True),
    (None, 'length', True), (2, 'length', True), (None, None, False),
])
def test_budget_flags(tokens, finish, reached):
    assert output_flags('FINAL: unknown', tokens, finish)['output_budget_reached'] is reached


def test_primary_scoring_retains_phase13_rules():
    assert score_answer('Mona Lisa', 'The Mona Lisa')
    assert not score_answer('103 years', '103')
    assert not score_answer('', '')


def test_no_truncation_and_full_output_budget():
    validate_budget(7936, 8192, 256)
    with pytest.raises(ValueError, match='truncation is forbidden'):
        validate_budget(7937, 8192, 256)


def test_tokenizer_batchencoding_is_supported():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{'role': 'user', 'content': 'question'}]
            assert kwargs == {'tokenize': True, 'add_generation_prompt': True, 'enable_thinking': False}
            return {'input_ids': [1, 2, 3]}
    assert chat_token_ids(Tokenizer(), 'question') == [1, 2, 3]


def evidence_fixture():
    url = 'https://en.wikipedia.org/wiki/Example'
    text = 'Original evidence. Additional unrelated text.'
    selection = {'url': url, 'excerpt_id': '1-B-1', 'char_start': 0,
                 'char_end': 18, 'text_sha256': sha256(text[:18].encode()).hexdigest()}
    texts = {url: {'text': text, 'archive_member': 'abc.html', 'html_sha256': 'html',
                   'extracted_text_sha256': 'text'}}
    evidence = [{'title': 'Example', 'url': url, 'text': text, 'passage_id': 'abc:0'}]
    old = {'sample_id': '1', 'question': 'What fact?', 'gold_answer': 'GOLD SHOULD NOT LEAK',
           'reference_urls': [url + '#heading'], 'retrieved_reference_count': 0,
           'stages': [{'stage': 'retrieve_rerank', 'output': {'evidence': evidence}}]}
    sample = {'sample_id': '1', 'verification': {'verifier': 'Codex agent', 'human_confirmed': False,
              'status': 'sufficient', 'reason': 'Required fact is present', 'required_facts': [{'assessment': 'supported'}]},
              'selections': [selection]}
    return {'samples': [sample]}, {'1': old}, texts


def test_verbatim_offsets_hash_and_reference_provenance():
    cfg, old, texts = evidence_fixture()
    original = deepcopy(old)
    row = phase15_contexts(cfg, old, texts)[0]
    assert row['b_excerpts'][0]['text'] == 'Original evidence.'
    assert row['b_excerpts'][0]['archive_member'] == 'abc.html'
    assert row['a_reference_hit_count_phase13_url_rule'] == 0
    assert row['a_reference_document_count_canonical'] == 1
    assert row['a_evidence_original'] == old['1']['stages'][0]['output']['evidence']
    assert old == original
    for key in ('a_context', 'b_context'):
        prompt = build_prompt(row['question'], row[key], 'Question: {question}\nEvidence: {context}')
        assert old['1']['gold_answer'] not in prompt


@pytest.mark.parametrize('change', ['hash', 'offset', 'reference', 'unverified', 'human', 'duplicate'])
def test_bad_or_mislabelled_evidence_is_rejected(change):
    cfg, old, texts = evidence_fixture()
    sample = cfg['samples'][0]
    if change == 'hash': sample['selections'][0]['text_sha256'] = 'modified'
    if change == 'offset': sample['selections'][0]['char_end'] = 9999
    if change == 'reference': old['1']['reference_urls'] = ['wrong']
    if change == 'unverified': sample['verification']['required_facts'][0]['assessment'] = 'missing'
    if change == 'human': sample['verification']['human_confirmed'] = True
    if change == 'duplicate': cfg['samples'].append(deepcopy(sample))
    with pytest.raises(ValueError): phase15_contexts(cfg, old, texts)


def test_outputs_cannot_overwrite_prior_file(tmp_path):
    path = tmp_path / 'phase15_output.json'
    exclusive_json(path, {'value': 1})
    with pytest.raises(FileExistsError): exclusive_json(path, {'value': 2})
    assert '"value": 1' in path.read_text()


def measured(sample_id, method, exact, status):
    return {'sample_id': sample_id, 'method': method, 'normalized_exact_match': exact,
            'final_answer_normalized': 'answer', 'has_final': True, 'output_budget_reached': False,
            'output_tokens': 2, 'input_tokens': 20, 'latency_ms': float(sample_id), 'context_chars': 10,
            'context_tokens': 2, 'b_verification_status': status, 'success': True,
            'output_format_valid': True, 'reference_document_count_canonical': 1,
            'nonreference_document_count_canonical': 0}


def paired_measurements():
    rows = []
    for sid, a, b, status in [('1', True, True, 'sufficient'), ('2', True, False, 'sufficient'),
                              ('3', False, True, 'sufficient'), ('4', False, False, 'sufficient'),
                              ('5', False, True, 'pending'), ('6', True, False, 'insufficient')]:
        rows += [measured(sid, METHODS[0], a, status), measured(sid, METHODS[1], b, status)]
    return rows


def test_pair_aggregation_excludes_pending_and_insufficient_without_dropping_tasks():
    summary = aggregate(paired_measurements())
    assert summary['generation_count'] == 12
    assert summary['question_count'] == 6
    assert summary['b_evidence_status_counts'] == {'sufficient': 4, 'insufficient': 1, 'pending': 1}
    pair = summary['paired_on_agent_verified_sufficient']
    assert pair['task_count'] == 4
    assert pair['human_confirmed_count'] == 0
    assert pair['outcomes'] == {'both_correct': 1, 'only_a_correct': 1, 'only_b_correct': 1, 'both_wrong': 1}
    assert summary['methods'][METHODS[0]]['exact_match_count'] == 3
    assert summary['methods'][METHODS[0]]['p95_latency_ms'] == pytest.approx(5.75)


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'unknown'])
def test_incomplete_or_duplicate_comparison_rejected(mutation):
    rows = paired_measurements()
    if mutation == 'duplicate': rows.append(rows[0])
    if mutation == 'missing': rows.pop()
    if mutation == 'unknown': rows[0]['method'] = 'other'
    with pytest.raises(ValueError): aggregate(rows)


def test_observation_uses_real_token_ids_finish_reason_and_identical_chat_settings():
    class Engine:
        def chat(self, messages, params, **kwargs):
            assert messages == [{'role': 'user', 'content': 'prompt'}]
            assert params is sampling
            assert kwargs['chat_template_kwargs'] == {'enable_thinking': False}
            return [SimpleNamespace(prompt_token_ids=[1, 2], outputs=[SimpleNamespace(
                text='FINAL: unknown', token_ids=[3, 4, 5], finish_reason='stop', stop_reason=None)])]
    sampling = object()
    backend = SimpleNamespace(_llm=Engine(), _sampling_params=sampling)
    item = {'request_id': '1-A', 'sample_id': '1', 'method': METHODS[0], 'prompt': 'prompt',
            'preflight_token_ids': [1, 2], 'preflight_input_tokens': 2}
    result = generate_one(backend, item)
    assert result['success']
    assert result['input_tokens'] == 2
    assert result['output_tokens'] == 3
    assert result['finish_reason'] == 'stop'
    assert result['final_answer'] == 'unknown'
    assert result['measurement_type'] == 'real_vllm_answer_only'
    item['preflight_token_ids'] = [1]
    result = generate_one(backend, item)
    assert not result['success']
    assert 'preflight' in result['error']


def collection_fixture(tmp_path):
    from scaleflow.baseline import file_sha256
    from scaleflow.phase15_diagnostics import exclusive_jsonl
    inputs, checks, generated = [], [], []
    for i in range(1, 25):
        sid = str(i)
        status = 'sufficient' if i == 1 else 'pending'
        checks.append({'sample_id': sid, 'question': 'What value?', 'gold_answer': '42',
                       'b_verification': {'status': status, 'reason': 'Fixture fact check'}})
        for group, method in zip(('A', 'B'), METHODS):
            request = f'{sid}-{group}'
            inputs.append({'request_id': request, 'sample_id': sid, 'method': method,
                           'preflight_input_tokens': 10, 'prompt_sha256': 'fixture',
                           'execution_index': len(inputs), 'context_tokens': 4, 'context_chars': 10,
                           'context_words': 2, 'reference_document_count_canonical': 1,
                           'nonreference_document_count_canonical': 0, 'context_truncated': False,
                           'b_verification_status': status})
            raw = 'FINAL: 41' if group == 'A' else 'FINAL: 42'
            generated.append({'request_id': request, 'sample_id': sid, 'method': method,
                              'input_tokens': 10, 'output_tokens': 3, 'raw_output': raw,
                              'finish_reason': 'stop', 'stop_reason': None, 'success': True,
                              'latency_ms': 5.0, **output_flags(raw, 3, 'stop')})
    ip = tmp_path / 'phase15_paired_inputs.jsonl'
    cp = tmp_path / 'phase15_evidence_checks.jsonl'
    rp = tmp_path / 'phase15_generations.jsonl'
    jp = tmp_path / 'phase15_job.json'
    exclusive_jsonl(ip, inputs)
    exclusive_jsonl(cp, checks)
    exclusive_jsonl(rp, generated)
    exclusive_json(jp, {'inputs_sha256': file_sha256(ip), 'evidence_checks_sha256': file_sha256(cp)})
    exclusive_json(tmp_path / 'phase15_generations.meta.json', {'job_sha256': file_sha256(jp), 'inputs_sha256': file_sha256(ip)})
    return generated


def test_semantic_candidates_cannot_change_exact_match_and_all_24_are_kept(tmp_path):
    from scaleflow.phase15_diagnostics import collect
    collection_fixture(tmp_path)
    review = tmp_path / 'reviews.json'
    exclusive_json(review, {'1-A': {'status': 'possible_equivalent', 'reason': 'Fixture candidate only'}})
    report = collect(tmp_path, review)
    assert report['methods'][METHODS[0]]['exact_match_count'] == 0
    assert report['methods'][METHODS[1]]['exact_match_count'] == 24
    assert report['paired_on_agent_verified_sufficient']['task_count'] == 1
    assert report['paired_on_agent_verified_sufficient']['outcomes']['only_b_correct'] == 1
    assert len((tmp_path / 'phase15_answer_measurements.jsonl').read_text().splitlines()) == 48
    assert len((tmp_path / 'phase15_paired_review.csv').read_text().splitlines()) == 25
    with pytest.raises(FileExistsError): collect(tmp_path, review)


@pytest.mark.parametrize('mutation', ['input_drift', 'flag_drift', 'missing_generation', 'prompt_token_drift', 'extra_review'])
def test_collection_checks_provenance_and_real_output_flags(tmp_path, mutation):
    import json
    from scaleflow.phase15_diagnostics import collect
    generated = collection_fixture(tmp_path)
    review = None
    if mutation == 'input_drift':
        with (tmp_path / 'phase15_paired_inputs.jsonl').open('a') as stream:
            stream.write('\n')
    else:
        if mutation == 'flag_drift': generated[0]['final_answer'] = '42'
        if mutation == 'missing_generation': generated.pop()
        if mutation == 'prompt_token_drift': generated[0]['input_tokens'] = 9
        if mutation == 'extra_review':
            review = tmp_path / 'reviews.json'
            exclusive_json(review, {'wrong-ID': {'status': 'exact_match', 'reason': 'fixture'}})
        (tmp_path / 'phase15_generations.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in generated))
    with pytest.raises(ValueError): collect(tmp_path, review)
