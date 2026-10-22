"""Dashboard labels accepted progress and never promotes failed paths."""
import json
from pathlib import Path
import tempfile
import unittest
from tools.render_solver_dashboard import case_state, render, shell


class SolverDashboardTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)

    def test_progress_uses_accepted_result_and_separates_trial(self):
        tmp_path=self.path
        (tmp_path/'result.json').write_text(json.dumps({'steps':[
            {'depth_mm':.025,'reaction_fz_n':.66,'residual_n':8e-5,'min_j':.91}]}))
        (tmp_path/'run.log').write_text('FRICTION_PTC_REJECT depth_mm=0.030 residual=9\n')
        state=case_state({'output':str(tmp_path),'name':'A0_base','status':'running','last_depth_mm':None})
        assert state['accepted_depth']==.025
        assert state['attempted_depth']==.030
        assert state['residual']==8e-5
        assert state['terminal_error'] is None

    def test_logs_and_errors_escaped_and_stale_worker_labelled(self):
        tmp_path=self.path
        job=dict(name='A0_base',tag='base',status='failed',output=str(tmp_path),
                 log_tail='<script>bad()</script>',terminal_error='<bad>',process=None)
        data=dict(queue={'status':'running'},cases=[job],worker={'matched':False,'reason':'进程已退出'},
                  observed_jst='test',report={},predecessor={},worker_log='',worker_stderr='')
        html=render(data,tmp_path)
        assert '<script>bad()' not in html
        assert '&lt;script&gt;bad()' in html
        assert '工作进程未确认' in html
        assert '完整求解完成' in html and '0 / 1 项' in html

    def test_static_page_does_not_claim_refresh(self):
        assert 'http-equiv="refresh"' not in shell('x','x')
        assert 'http-equiv="refresh"' in shell('x','x',True)

    def test_any_angle_queue_uses_vertical_travel_without_old_study_controls(self):
        tmp_path=self.path
        data=dict(queue={'status':'finished'},cases=[],worker={'matched':False,'reason':'进程已退出'},
                  observed_jst='test',worker_log='',worker_stderr='')
        html=render(data,tmp_path)
        assert '竖直刀具行程' in html
        assert '15 条' not in html and '法向 q' not in html

if __name__=='__main__':unittest.main()
