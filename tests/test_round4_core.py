"""Core round-4 regressions; evaluator artifacts stay in temporary workspaces."""
import hashlib
import io
import json
import os
from contextlib import redirect_stdout
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from autodata.cs import run_paper
from autodata.cs.evaluate_rubric import legacy_question_hash, main
from autodata.cs.parsing import parse_qv_output
from autodata.cs.prompts import PromptSet
from autodata.cs.run_paper import PaperInput, PaperRun, RoundRecord, _bind_norm, final_filter, prepare_workspace
from autodata.llm.client import LLMClient
from tests.fake_openai_server import ScriptedResponder, make_transport, text_response
from tests.test_e2e_fake import EVAL_INPUT, PROMPTS, QV_PASS, Scenario, _config, _role


@pytest.fixture
def evaluator_run(tmp_path, monkeypatch):
    cfg = _config('http://fake.invalid/v1', tmp_path)
    cfg.eval.n_attempts = 1
    qv = ScriptedResponder([text_response(QV_PASS)])
    clients = {'quality_verifier': LLMClient(cfg.endpoint('quality_verifier'), transport=make_transport(qv))}
    run = PaperRun(cfg, PaperInput('paper', 'Title', 'body ' * 100), tmp_path / 'paper', clients,
                   PromptSet(PROMPTS), PROMPTS, log=lambda _: None)
    run.ws = prepare_workspace(cfg, run.paper, run.workdir, PROMPTS)
    run.api_config_sha1 = hashlib.sha1((run.workdir / run_paper.API_CONFIG_REL).read_bytes()).hexdigest()
    candidate = deepcopy(EVAL_INPUT)
    run.rounds = [RoundRecord(1, 0, challenger_json=deepcopy(candidate), challenger_output=json.dumps(candidate))]
    (run.workdir / 'eval_input.json').write_text(json.dumps(candidate))
    state = {'weak_pass': True, 'strong_pass': True, 'modes': [], 'qv': qv}
    scenario = Scenario()

    def respond(body, index):
        if _role(body) == 'judge':
            strong = 'STRONG ANSWER' in body['messages'][-1]['content']
            if (strong and not state['strong_pass']) or (not strong and not state['weak_pass']):
                flags = [False] * 11 if strong else [True] * 8 + [False] * 3
                return text_response(json.dumps({'criteria': [
                    {'index': i, 'satisfied': flag, 'evidence': ''} for i, flag in enumerate(flags, 1)]}))
        return scenario(body, index)

    async def evaluate(workdir, argv_abs, *, deadline_s):
        state['modes'].append('weak-only' if '--weak-only' in argv_abs else
                              'strong-only' if '--strong-only' in argv_abs else 'both')
        output = io.StringIO()
        with redirect_stdout(output):
            rc = main(argv_abs, transport=make_transport(respond))
        return output.getvalue(), '', rc

    monkeypatch.setattr(run_paper, 'run_evaluator', evaluate)
    return run, state


@pytest.mark.parametrize('first,requested,weak_pass,strong_pass,expected_modes,cached', [
    ('both', 'weak-only', True, True, ['both'], True),
    ('both', 'strong-only', True, True, ['both'], True),
    ('both', 'both', True, True, ['both'], True),
    ('weak-only', 'weak-only', True, True, ['weak-only'], True),
    ('weak-only', 'both', True, True, ['weak-only', 'strong-only'], False),
    ('weak-only', 'both', False, True, ['weak-only'], True),
    ('both', 'weak-only', False, True, ['both'], True),
    ('both', 'both', False, True, ['both'], True),
    ('both', 'both', True, False, ['both'], True),
    ('both', 'strong-only', True, False, ['both'], True),
])
async def test_stage_cache(evaluator_run, first, requested, weak_pass, strong_pass, expected_modes, cached):
    run, state = evaluator_run
    state.update(weak_pass=weak_pass, strong_pass=strong_pass)
    argv = lambda mode: [] if mode == 'both' else ['--' + mode]
    first_output = await run.run_evaluate_rubric(argv(first))
    assert run.rounds[0].evals[-1]['verified']
    # Equivalent JSON layout and trailing whitespace inside strings share an identity.
    variant = deepcopy(EVAL_INPUT)
    variant['context'] = variant['context'].rstrip() + '\t '
    variant['question'] += '  '
    variant['rubric'] = [dict(reversed(list(item.items()))) for item in variant['rubric']]
    (run.workdir / 'eval_input.json').write_text(json.dumps(variant, sort_keys=True, indent=3))
    output = await run.run_evaluate_rubric(argv(requested))
    assert state['modes'] == expected_modes
    assert len(list((run.workdir / 'eval_attempts').glob('run_*'))) == len(expected_modes)
    if cached:
        assert output.startswith(first_output)
        assert '[cached:' in output
    else:
        assert 'executed --strong-only' in output
        assert run.rounds[0].evals[-1]['verified'], run.rounds[0].evals[-1]['problems']
        assert run.rounds[0].evals[-1]['report']['weak_source_report'] == run.rounds[0].evals[0]['report_path']


async def test_split_stages_cache_both_and_legacy_harness_verification(evaluator_run):
    run, state = evaluator_run
    await run.run_evaluate_rubric(['--weak-only'])
    weak = run.rounds[0].evals[-1]
    weak['report']['question_hash'] = legacy_question_hash(weak['candidate'])
    from pathlib import Path
    Path(weak['report_path']).write_text(json.dumps(weak['report']))
    verdict, problems = run._verify_report(weak['report'], weak['candidate'], weak['input_sha1'], 0,
                                           Path(weak['report_path']), {Path(weak['report_path']).parent}, mode='weak-only')
    assert not problems and verdict['weak_passed']
    await run.run_evaluate_rubric(['--strong-only'])
    assert run.rounds[0].evals[-1]['verified']
    assert '[cached:' in await run.run_evaluate_rubric([])
    assert state['modes'] == ['weak-only', 'strong-only']


async def test_unverified_evaluation_is_not_cached(evaluator_run):
    run, state = evaluator_run
    await run.run_evaluate_rubric(['--weak-only'])
    run.rounds[0].evals[-1]['verified'] = False
    await run.run_evaluate_rubric(['--weak-only'])
    assert state['modes'] == ['weak-only', 'weak-only']


@pytest.mark.parametrize('original,edited,parsed,allowed', [
    ('What if x < 0?', 'What if x > 0?', True, False),
    ('Compare α and γ.', 'Compare β and γ.', True, False),
    ('Use X.', 'Use x.', True, False),
    ('Why "café"?\nExplain.', 'Why "cafe\u0301"?  \r\n Explain.', True, True),
    ('What if x < 0?', 'What if x > 0?', False, False),
    ('Compare α and γ.', 'Compare β and γ.', False, False),
    ('Why now?\nExplain.', 'Why now?  Explain.', False, True),
])
async def test_challenger_origin_preserves_operators_case_and_unicode(evaluator_run, original, edited, parsed, allowed):
    run, state = evaluator_run
    rec = run.rounds[0]
    rec.challenger_json = {**deepcopy(EVAL_INPUT), 'question': original} if parsed else None
    rec.challenger_output = 'Malformed challenger object:\nquestion: ' + original + '\n rubric: ...'
    (run.workdir / 'eval_input.json').write_text(json.dumps({**EVAL_INPUT, 'question': edited}, ensure_ascii=True))
    output = await run.run_evaluate_rubric(['--weak-only'])
    assert bool(state['modes']) is allowed
    assert ('not the challenger' in output) is not allowed


def binding_candidate():
    return {'context': 'The context of this case.', 'question': 'Why "café" and α?',
            'rubric': [{'criterion': 'Explain the complete derivation ' + 'with justified intermediate steps ' * 8,
                        'weight': 8, 'category': 'positive'}]}


@pytest.mark.parametrize('format', ['json-after', 'json-before', 'numbered-bullet'])
@pytest.mark.parametrize('weight', [8, -5])
def test_binding_long_criteria_formats_and_escapes(format, weight):
    candidate = binding_candidate()
    item = candidate['rubric'][0]
    item.update(weight=weight, category='positive' if weight > 0 else 'negative')
    if format == 'json-before':
        candidate['rubric'] = [dict(weight=weight, criterion=item['criterion'], category=item['category'])]
    if format.startswith('json'):
        prompt = json.dumps(candidate, ensure_ascii=True)
    else:
        prompt = f"{candidate['question']}\n{candidate['context']}\n1. [{weight:+d}] {item['criterion']}"
    assert PaperRun._qv_missing(prompt, candidate) == []
    rec = RoundRecord(1, 0, qv_calls=[{'prompt': prompt}])
    assert PaperRun.__new__(PaperRun)._qv_bound(rec, candidate)  # no workspace needed


@pytest.mark.parametrize('original,quoted', [(8, -8), (-5, 5), (1, 10), (-1, -10), (8, 18), (-5, -15)])
def test_binding_rejects_sign_and_magnitude_changes(original, quoted):
    candidate = binding_candidate()
    candidate['rubric'][0]['weight'] = original
    prompt = f"{candidate['question']} {candidate['context']} {quoted:+d} {candidate['rubric'][0]['criterion']}"
    assert PaperRun._qv_missing(prompt, candidate) == ['weight of criterion 1']


def test_binding_rejects_replaced_criterion_and_accepts_quoted_prefix():
    candidate = binding_candidate()
    prefix = _bind_norm(candidate['rubric'][0]['criterion'])[:80]
    header = f"{candidate['question']} {candidate['context']} "
    assert PaperRun._qv_missing(header + '8 ' + prefix, candidate) == []
    assert PaperRun._qv_missing(header + '8 An unrelated criterion.', candidate) == ['criterion 1']


def test_binding_normalisation_keeps_digit_separators_and_question_fallback():
    assert _bind_norm('1. [-5] text') == '1 -5 text'
    assert _bind_norm('"weight": -5,') == 'weight -5'
    assert _bind_norm('"weight": +8') == 'weight 8'
    candidate = binding_candidate()
    prompt = json.dumps(candidate, ensure_ascii=False).replace('Why', 'WHY')
    assert not PaperRun._qv_missing(prompt, candidate)


@pytest.mark.parametrize('omitted,missing', [('question', 'question'), ('context', 'context head'),
                                           ('rubric', 'criterion 1')])
async def test_completed_qv_repeat_depends_on_quoted_question_only(evaluator_run, omitted, missing):
    run, state = evaluator_run
    incomplete = {k: v for k, v in EVAL_INPUT.items() if k != omitted}
    text = await run.run_subagent('quality_verifier', 'incomplete', json.dumps(incomplete))
    assert text == QV_PASS
    diagnostic = run._round_summary(run.rounds[0])
    assert diagnostic['qv_bound'] is False and missing in diagnostic['qv_missing']
    text = await run.run_subagent('quality_verifier', 'complete', json.dumps(EVAL_INPUT))
    if omitted == 'question':
        assert text == QV_PASS
        diagnostic = run._round_summary(run.rounds[0])
        assert diagnostic['qv_bound'] is True and diagnostic['qv_missing'] == []
    else:
        assert 'already completed' in text
    text = await run.run_subagent('quality_verifier', 'repeat', json.dumps(EVAL_INPUT))
    assert 'already completed' in text
    assert len(state['qv'].requests) == (2 if omitted == 'question' else 1)


@pytest.mark.parametrize('bound', [False, True])
async def test_passing_qv_binding_is_informational_at_acceptance(evaluator_run, bound):
    run, state = evaluator_run
    prompt = json.dumps(EVAL_INPUT) if bound else EVAL_INPUT['question']
    assert await run.run_subagent('quality_verifier', 'review', prompt) == QV_PASS
    output = await run.run_evaluate_rubric([])
    assert 'Acceptance refused' not in output
    assert run.accepted and state['modes'] == ['both']
    missing = PaperRun._qv_missing(prompt, EVAL_INPUT)
    assert run.accepted['qv_bound'] is bound and run.accepted['qv_missing'] == missing
    diagnostic = run._round_summary(run.rounds[0])
    assert diagnostic['qv_bound'] is bound and diagnostic['qv_missing'] == missing
    events = [event for event in run.guardrail_events if event['kind'] == 'qv_not_bound']
    assert len(events) == (0 if bound else 1)
    assert all(event['informational'] for event in events)


async def test_out_of_range_weights_are_nonblocking_warnings(evaluator_run):
    run, state = evaluator_run
    candidate = deepcopy(EVAL_INPUT)
    for item in candidate['rubric']:
        item['weight'] *= 3
    (run.workdir / 'eval_input.json').write_text(json.dumps(candidate))
    run.rounds[0].challenger_json = candidate
    await run.run_subagent('quality_verifier', 'complete', json.dumps(candidate))
    await run.run_evaluate_rubric([])
    ev = run.rounds[0].evals[-1]
    assert ev['verified'] and ev['problems'] == [] and ev['warnings']
    assert run._round_summary(run.rounds[0])['eval_warnings'] == ev['warnings']
    assert run.accepted is not None
    assert any(e['kind'] == 'rubric_weights_out_of_spec' and e['informational'] for e in run.guardrail_events)
    filtered = final_filter(run.cfg, candidate['context'], candidate['rubric'])
    assert filtered['weights_in_spec'] is False and filtered['rubric_ok'] is True


@pytest.mark.parametrize('separator', [' | ', ' or ', '/'])
@pytest.mark.parametrize('format', ['OVERALL: {}', '- **OVERALL:** **{}**', 'The OVERALL => {}'])
def test_qv_echoed_overall_alternatives_unresolved(separator, format):
    text = QV_PASS.replace('OVERALL: PASS', format.format('PASS' + separator + 'FAIL'))
    parsed = parse_qv_output(text)
    assert parsed['overall_stated'] is None and parsed['overall'] is None


@pytest.mark.parametrize('separator', [' | ', ' or ', '/'])
def test_qv_echoed_check_is_missing(separator):
    text = QV_PASS.replace('CHECK_1_VERDICT: NO_LEAKAGE', 'CHECK_1_VERDICT: NO_LEAKAGE' + separator + 'LEAKAGE')
    parsed = parse_qv_output(text)
    assert parsed['missing_checks'] == ['CHECK_1_VERDICT']
    assert parsed['overall'] is False


def test_qv_fallback_last_standalone_verdict():
    text = QV_PASS.replace('OVERALL: PASS', 'The OVERALL => FAIL\nThe OVERALL => PASS')
    assert parse_qv_output(text)['overall'] is True


async def test_evaluator_passes_expected_parent_pid(tmp_path, monkeypatch):
    process = AsyncMock()
    process.returncode = 0
    process.communicate.return_value = (b'ok', b'')
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(run_paper.asyncio, 'create_subprocess_exec', spawn)
    monkeypatch.setenv('AUTODATA_PARENT_PID', 'stale')
    assert await run_paper.run_evaluator(tmp_path, [], deadline_s=5) == ('ok', '', 0)
    assert spawn.call_args.kwargs['env']['AUTODATA_PARENT_PID'] == str(os.getpid())


@pytest.mark.parametrize('alternative', ['TOO_EASY', 'RECALL'])
def test_qv_actual_difficulty_template_is_unresolved(alternative):
    text = QV_PASS.replace('CHECK_2_VERDICT: GOOD', 'CHECK_2_VERDICT: GOOD | ' + alternative)
    parsed = parse_qv_output(text)
    assert parsed['missing_checks'] == ['CHECK_2_VERDICT'] and parsed['overall'] is False


async def test_unbound_qv_reports_missing_signed_weight(evaluator_run):
    run, state = evaluator_run
    prompt = deepcopy(EVAL_INPUT)
    prompt['rubric'][0]['weight'] = -5
    # Other positive fives must be outside the first criterion's weight window.
    prompt['rubric'][0]['criterion'] = 'Explain ' + 'a necessary intermediate argument ' * 8
    run.rounds[0].challenger_json['rubric'][0]['criterion'] = prompt['rubric'][0]['criterion']
    output = await run.run_subagent('quality_verifier', 'wrong sign', json.dumps(prompt))
    assert output == QV_PASS
    diagnostic = run._round_summary(run.rounds[0])
    assert diagnostic['qv_bound'] is False and 'weight of criterion 1' in diagnostic['qv_missing']


async def test_qv_failed_but_bound_cannot_be_repeated(evaluator_run):
    run, state = evaluator_run
    state['qv'].responses = [text_response(QV_PASS.replace('OVERALL: PASS', 'OVERALL: FAIL'))]
    await run.run_subagent('quality_verifier', 'first', json.dumps(EVAL_INPUT))
    output = await run.run_subagent('quality_verifier', 'again', json.dumps(EVAL_INPUT))
    assert 'already completed' in output and 'OVERALL: FAIL' in output
    assert len(state['qv'].requests) == 1


@pytest.mark.parametrize('rubric', [[], None, [{'criterion': 'Explain.', 'weight': 0}],
                                   [{'criterion': 'Explain.', 'weight': 5, 'category': 'negative'}]])
async def test_invalid_rubric_returns_input_error_without_evaluator(evaluator_run, rubric):
    run, state = evaluator_run
    (run.workdir / 'eval_input.json').write_text(json.dumps({**EVAL_INPUT, 'rubric': rubric}))
    output = await run.run_evaluate_rubric([])
    assert output.startswith('INPUT_ERROR: invalid question or rubric')
    assert state['modes'] == [] and not run.accepted
    assert run.guardrail_events[-1]['kind'] == 'bad_eval_input'


async def test_round_without_qv_records_missing_elements(evaluator_run):
    run, _ = evaluator_run
    summary = run._round_summary(run.rounds[0])
    assert summary['qv_bound'] is False
    assert summary['qv_missing'] == PaperRun._qv_missing('', EVAL_INPUT)


async def test_unbound_completed_qv_repeat_refused_with_escaped_question(evaluator_run):
    run, state = evaluator_run
    question = 'Why "café" and α?'
    run.rounds[0].challenger_json['question'] = question
    await run.run_subagent('quality_verifier', 'first', json.dumps({'question': question}, ensure_ascii=True))
    output = await run.run_subagent('quality_verifier', 'repeat', 'Please check again.')
    assert 'already completed' in output and len(state['qv'].requests) == 1


async def test_incomplete_qv_can_be_retried(evaluator_run):
    run, state = evaluator_run
    state['qv'].responses = [text_response(QV_PASS, finish_reason='length'), text_response(QV_PASS)]
    await run.run_subagent('quality_verifier', 'incomplete', EVAL_INPUT['question'])
    assert run.rounds[0].qv_calls[-1]['stop_reason'] == 'length'
    assert await run.run_subagent('quality_verifier', 'retry', json.dumps(EVAL_INPUT)) == QV_PASS
    assert len(state['qv'].requests) == 2
