"""Presentation must not promote attempted steps or pending gates to results."""
import json
from pathlib import Path

from tools.render_solver_dashboard import analysis_section, case_state, render, shell


def test_progress_uses_accepted_result_and_separates_trial(tmp_path):
    (tmp_path/'result.json').write_text(json.dumps({'steps':[
        {'depth_mm':.025,'reaction_fz_n':.66,'residual_n':8e-5,'min_j':.91}]}))
    (tmp_path/'run.log').write_text('FRICTION_PTC_REJECT depth_mm=0.030 residual=9\n')
    state=case_state({'output':str(tmp_path),'name':'A0_base','status':'running','last_depth_mm':None})
    assert state['accepted_depth']==.025
    assert state['attempted_depth']==.030
    assert state['residual']==8e-5
    assert state['terminal_error'] is None


def test_pending_controls_are_not_a_failed_gate(tmp_path):
    data={'report':{'joint_gate_passed':False},'queue':{'status':'running'},
          'cases':[{'tag':'halfstep','status':'pending'}]*15}
    html=analysis_section(data,tmp_path)
    assert '尚待验证' in html and '目前不能解释为已判定失败' in html


def test_logs_and_errors_escaped_and_stale_worker_labelled(tmp_path):
    job=dict(name='A0_base',tag='base',status='failed',output=str(tmp_path),
             log_tail='<script>bad()</script>',terminal_error='<bad>',process=None)
    data=dict(queue={'status':'running'},cases=[job],worker={'matched':False,'reason':'进程已退出'},
              observed_jst='test',report={},predecessor={},worker_log='',worker_stderr='')
    html=render(data,tmp_path)
    assert '<script>bad()' not in html
    assert '&lt;script&gt;bad()' in html
    assert '工作进程未确认' in html
    assert '完整求解完成' in html and '0 / 1 项' in html


def test_static_page_does_not_claim_refresh():
    assert 'http-equiv="refresh"' not in shell('x','x')
    assert 'http-equiv="refresh"' in shell('x','x',True)


def test_failed_primary_not_hidden_by_pending_controls(tmp_path):
    data={'report':{'joint_gate_passed':False},'queue':{'status':'running'},
          'cases':[{'name':'C57_base','tag':'base','status':'budget_exhausted',
                    'mechanics_completed':False,'evidence_verified':True}]+
                  [{'tag':'halfstep','status':'pending'}]*15}
    html=analysis_section(data,tmp_path)
    assert 'C57_base 已终结但未完成' in html
    assert 'false 目前不能解释为已判定失败' not in html
