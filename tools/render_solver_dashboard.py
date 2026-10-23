"""Read-only HTML for any bounded material queue; no mechanics imports or solves."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from html import escape
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import quote

import psutil

ROOT = Path(__file__).resolve().parents[1]
JST = timezone(timedelta(hours=9))
ACTIVE = {'running', 'prepared', 'waiting_for_predecessor'}
LABELS = {'running': '运行中', 'pending': '等待', 'prepared': '已准备',
          'waiting_for_predecessor': '等待前置任务', 'completed': '完成',
          'finished': '队列结束', 'failed_or_unverified': '失败或证据未通过',
          'timeout': '超时', 'budget_exhausted': '预算耗尽（未完成）',
          'failed': '失败', 'skipped': '跳过'}
CSS = '''
:root{color-scheme:light;--ink:#182c40;--muted:#53677a;--line:#d8e2ea;--blue:#086b84}
*{box-sizing:border-box}body{margin:0;background:#f2f6f9;color:var(--ink);font:16px/1.65 "Segoe UI","Microsoft YaHei",sans-serif}
main{max-width:1250px;margin:auto;padding:32px 24px 60px}h1{font-size:30px;margin:8px 0}h2{font-size:22px;margin-top:0}h3{font-size:18px}
a{color:#006b86}header{margin-bottom:24px}nav{display:flex;gap:20px;flex-wrap:wrap}.sub{color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px;margin:20px 0}
.card,section{background:white;border:1px solid var(--line);border-radius:12px;padding:22px;margin-bottom:20px}
.card{margin:0}.value{display:block;font-size:25px;font-weight:650;margin:8px 0;color:var(--blue)}
.notice{border-left:4px solid #b66b10;background:#fff6e8;padding:14px 18px;margin:18px 0}
.ok{color:#16734a}.bad{color:#a13022}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:10px;border-bottom:1px solid var(--line);vertical-align:top}th{background:#edf3f7;white-space:nowrap}
pre{background:#102437;color:#e3eef7;padding:18px;max-height:420px;overflow:auto;font:12px/1.65 Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere}
code{font-size:.9em;overflow-wrap:anywhere}summary{cursor:pointer;padding:10px 0;font-weight:600}details{margin:12px 0}
progress{width:100%;accent-color:var(--blue);height:12px}.pill{font-size:13px;padding:4px 10px;border-radius:20px;background:#e3eff4}
footer{font-size:13px;color:var(--muted)}.document img{max-width:100%}.document blockquote{border-left:3px solid var(--line);margin-left:0;padding-left:18px}
@media(max-width:650px){main{padding:16px 12px}h1{font-size:25px}section,.card{padding:16px}}
'''


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as exc:
        return {'_read_error': f'{type(exc).__name__}: {exc}'}


def tail(path, lines=28):
    try:
        with Path(path).open('rb') as f:
            f.seek(max(0, f.seek(0, 2) - 32768))
            return '\n'.join(f.read().decode('utf-8', errors='replace').splitlines()[-lines:])
    except OSError:
        return ''


def stamp(value=None):
    try:
        dt = datetime.fromisoformat(value) if value else datetime.now(timezone.utc)
        return dt.astimezone(JST).strftime('%Y-%m-%d %H:%M:%S JST')
    except (TypeError, ValueError):
        return '未记录'


def number(value, digits=5):
    return f'{value:.{digits}g}' if isinstance(value, (float, int)) else '—'


def process_check(pid, command, started):
    """PID alone is never treated as a live solver identity."""
    if not pid or not command or not started:
        return {'matched': None, 'reason': '缺少进程身份资料'}
    try:
        p = psutil.Process(pid)
        expected = datetime.fromisoformat(started).timestamp()
        matches = p.cmdline() == command and abs(p.create_time() - expected) < 120
        return {'matched': matches, 'pid': pid,
                'reason': '命令与创建时间匹配' if matches else '进程身份不匹配'}
    except psutil.NoSuchProcess:
        return {'matched': False, 'pid': pid, 'reason': '进程已退出'}
    except psutil.Error as exc:
        return {'matched': None, 'pid': pid, 'reason': str(exc)}


def case_state(job):
    folder = Path(job['output'])
    result = read_json(folder/'result.json')
    steps = result.get('steps', [])
    last = steps[-1] if steps else {}
    log = tail(folder/'run.log')
    attempted = re.findall(r'\bdepth_mm=([\d.eE+-]+)', log)
    attempted_depth = float(attempted[-1]) if attempted and job['status'] == 'running' else None
    try:
        updated = datetime.fromtimestamp((folder/'run.log').stat().st_mtime, timezone.utc)
        log_stamp = stamp(updated.isoformat())
        log_age = max(0, time.time()-updated.timestamp())
    except OSError:
        log_stamp, log_age = '尚无日志', None
    return dict(job, accepted_depth=last.get('depth_mm', job.get('last_depth_mm')),
                fz=last.get('reaction_fz_n', job.get('endpoint_fz_n')),
                residual=last.get('residual_n'), min_j=last.get('min_j'),
                state_count=len(steps), attempted_depth=attempted_depth,
                log_tail=log, log_updated=log_stamp, log_age_s=log_age,
                stderr_tail=tail(folder.parent/(job['name']+'.stderr.log')),
                terminal_error=result.get('error') or job.get('error'),
                result_read_error=result.get('_read_error') if job['status'] != 'pending' else None,
                process=process_check(job.get('solver_pid'), job.get('command'), job.get('started_utc'))
                if job['status'] == 'running' else None)


def collect(study):
    q = read_json(study/'queue_status.json')
    launch = read_json(study/'launch.json')
    return dict(observed_jst=stamp(), study=str(study), queue=q,
                cases=[case_state(j) for j in q.get('jobs', [])],
                worker=process_check(launch.get('worker_pid'), launch.get('command'), launch.get('started_utc')),
                worker_stderr=tail(study/'worker.stderr.log'),
                worker_log=tail(study/'worker.stdout.log', 12))


def link(path, label, root):
    href = quote(os.path.relpath(path, root).replace('\\', '/'), safe='/.:')
    return f'<a href="{escape(href, quote=True)}">{escape(label)}</a>'


def shell(title, body, refresh=False):
    auto = '<meta http-equiv="refresh" content="60">' if refresh else ''
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'{auto}<title>{escape(title)}</title><style>{CSS}</style></head><body><main>{body}'
            '</main></body></html>')


def job_table(cases, study):
    rows = []
    for j in cases:
        depth = j.get('accepted_depth', j.get('last_depth_mm'))
        force = j.get('fz', j.get('endpoint_fz_n'))
        verified = '终态复核通过' if j.get('mechanics_completed') and j.get('fresh_audit_passed') else ('待终态复核' if j['status'] in ACTIVE|{'pending','starting'} else '未通过 / 未提供')
        status = LABELS.get(j['status'], j['status'])
        if j.get('process') and j['process']['matched'] is not True:
            status += '；进程未确认'
        title = (f'<a href="#log-{escape(j["name"], quote=True)}">{escape(j["name"])}</a>'
)
        rows.append(f'<tr><td>{title}</td><td>{escape(status)}</td><td>{number(depth)}</td>'
                    f'<td>{number(force,7)}</td><td>{escape(verified)}</td></tr>')
    coordinate = '竖直刀具行程 d'
    return ('<div class="table"><table><thead><tr><th>案例 / 原始日志</th><th>队列状态</th>'
            f'<th>已接受 {coordinate} / mm</th><th>Fz / N</th><th>证据状态</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div>')


def render(data, study, refresh=False):
    q=data['queue'];cases=data['cases']
    done=sum(j.get('mechanics_completed') is True for j in cases)
    status=LABELS.get(q.get('status'),q.get('status','状态文件不可读'))
    if q.get('status') in ACTIVE and data['worker']['matched'] is not True:
        status+='（仅文件声明；工作进程未确认）'
    body=f'<header><h1>橡胶斜切 · 运行状态</h1><p>快照：{escape(data["observed_jst"])} · {escape(study.name)}</p></header>'
    body+=f'<section><h2>{escape(status)}</h2><p>完整求解完成：{done} / {len(cases)} 项。</p>'
    body+='<p>深度统一为实验室竖直刀具行程 d。只显示接受状态；尝试深度不计为已完成。完整体计算不预测断裂。</p>'
    body+=job_table(cases,study)+'</section>'
    if q.get('experiment_comparison'):
        body+='<p>'+link(study/q['experiment_comparison'],'实验对照、分段斜率与角度差',study)+'</p>'
    for j in cases:
        name=escape(j['name']);target=j.get('target_mm',q.get('target_mm',2))
        body+=f'<section id="log-{name}"><h2>{name}</h2><p>已接受 {number(j.get("accepted_depth"))} / {number(target)} mm；{j.get("state_count",0)} 状态。'
        body+=f' Fz={number(j.get("fz"),8)} N，残差={number(j.get("residual"),6)} N，min J={number(j.get("min_j"),6)}。</p>'
        if j.get('attempted_depth') is not None:body+=f'<p>最新尝试深度：{number(j["attempted_depth"])} mm。</p>'
        if j.get('process'):body+='<p>'+escape(j['process']['reason'])+'</p>'
        if j.get('log_updated'):body+='<p>日志更新：'+escape(j['log_updated'])+'</p>'
        body+='<details'+(' open' if j['status']=='running' else '')+'><summary>最近日志</summary><pre>'+escape(j.get('log_tail') or '尚无日志')+'</pre></details>'
        for key in ('terminal_error','result_read_error','stderr_tail'):
            if j.get(key):body+='<pre>'+escape(str(j[key]))+'</pre>'
        body+=link(Path(j['output'])/'run.log','完整运行日志',study)+' · '+link(Path(j['output'])/'result.json','接受记录',study)+'</section>'
    body+='<section><h2>调度日志</h2><pre>'+escape(data.get('worker_log') or '暂无')+'</pre><pre>'+escape(data.get('worker_stderr') or '标准错误为空')+'</pre>'
    body+=link(study/'queue_status.json','机器可读队列状态',study)+'</section>'
    body+='<footer>'+('后台报告定期刷新；页面每 60 秒重载。' if refresh else '本页为静态快照。')
    body+='时间戳决定数据新鲜度；未完成或未经整路径复核的结果不标为完整成功。工作进程：'+escape(data['worker']['reason'])+'</footer>'
    return shell('橡胶斜切 · 运行状态',body,refresh)


def atomic_write(path, text):
    from run_support import replace_with_retry
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(text, encoding='utf-8')
    replace_with_retry(temp, path)


def render_document(source, study):
    import markdown
    content = markdown.markdown(source.read_text(encoding='utf-8'), extensions=['tables','fenced_code','sane_lists'])
    # Preserve relative links from the source document's directory.
    def relocate(match):
        url = match.group(2)
        if re.match(r'^(?:[a-zA-Z]+:|#|/)', url):
            return match.group(0)
        absolute = os.path.relpath(source.parent/url.split('#')[0], study).replace('\\','/')
        fragment = '#'+url.split('#',1)[1] if '#' in url else ''
        return match.group(1)+'"'+quote(absolute, safe='/.:')+fragment+'"'
    content = re.sub(r'(href=|src=)"([^"]+)"', relocate, content)
    intro = '<nav><a href="index.html">返回运行状态</a></nav><p class="notice">理论与方案文档的 HTML 展示。文中启动时间是历史记录；实时进展以状态页快照为准。</p>'
    atomic_write(study/'analysis.html', shell('角度力学机理与试验设计', intro+'<section class="document">'+content+'</section>'))


def write_snapshot(study, refresh=False):
    data = collect(study)
    atomic_write(study/'dashboard_snapshot.json', json.dumps(data, ensure_ascii=False, indent=2))
    atomic_write(study/'index.html', render(data, study, refresh))
    return data


def watch(study, hours):
    lock = study/'dashboard.lock'
    # Exclusive lock prevents duplicate reporters. Existing lock is never stolen.
    with lock.open('x', encoding='utf-8') as f:
        json.dump({'pid': os.getpid(), 'started_jst': stamp()}, f)
    try:
        deadline, missing = time.monotonic()+hours*3600, 0
        while True:
            data = write_snapshot(study, refresh=True)
            print(stamp(), data['queue'].get('status'), flush=True)
            missing = missing+1 if data['worker']['matched'] is False else 0
            if data['queue'].get('status') not in ACTIVE or missing >= 3 or time.monotonic() >= deadline:
                write_snapshot(study, refresh=False)
                break
            time.sleep(60)
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--report-md', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--watch', action='store_true')
    mode.add_argument('--launch-watch', action='store_true')
    parser.add_argument('--max-hours', type=float, default=24)
    a = parser.parse_args(); study = a.study.resolve()
    if not (study/'queue_status.json').is_file() or not 0 < a.max_hours <= 48:
        parser.error('Require existing queue_status.json and 0 < max-hours <= 48')
    if a.report_md:
        render_document(a.report_md.resolve(), study)
    if a.launch_watch:
        if (study/'dashboard.lock').exists():
            parser.error('A reporter lock exists; inspect it before relaunching')
        write_snapshot(study)
        command = [sys.executable,'-X','utf8','-u','-m','tools.render_solver_dashboard',
                   '--study',str(study),'--watch','--max-hours',str(a.max_hours)]
        options = {}
        if os.name == 'nt':
            startup = subprocess.STARTUPINFO();startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = subprocess.SW_HIDE
            options = dict(startupinfo=startup, creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
        with (study/'dashboard.stdout.log').open('ab') as stdout, (study/'dashboard.stderr.log').open('ab') as stderr:
            process = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                       start_new_session=os.name!='nt',**options)
        launch = dict(pid=process.pid, command=command, started_utc=datetime.now(timezone.utc).isoformat())
        atomic_write(study/'dashboard_launch.json',json.dumps(launch,indent=2))
        print(json.dumps(launch))
    elif a.watch:
        watch(study,a.max_hours)
    else:
        data = write_snapshot(study)
        print(json.dumps({'html':str(study/'status.html'),'snapshot':data['observed_jst']},ensure_ascii=False))


if __name__ == '__main__':
    main()
