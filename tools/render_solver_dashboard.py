"""Read-only HTML for the matched normal-history study; never imports a solver.

python -m tools.render_solver_dashboard --study outputs/angle_paths_20261018
Add --watch for a bounded reporting-only process, or --launch-watch to detach it.
"""
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
    manifest = read_json(study/'manifest.json')
    predecessor = manifest.get('predecessor', {}).get('output')
    return dict(observed_jst=stamp(), study=str(study), queue=q,
                cases=[case_state(j) for j in q.get('jobs', [])],
                worker=process_check(launch.get('worker_pid'), launch.get('command'), launch.get('started_utc')),
                report=read_json(study/'study_report.json'),
                predecessor=read_json(Path(predecessor)/'queue_status.json') if predecessor else {},
                predecessor_path=predecessor, worker_stderr=tail(study/'worker.stderr.log'),
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


def job_table(cases, study, normal=True):
    rows = []
    for j in cases:
        depth = j.get('accepted_depth', j.get('last_depth_mm'))
        force = j.get('fz', j.get('endpoint_fz_n'))
        verified = ('终态复核通过' if j.get('mechanics_completed') else '已接受前缀复核通过') if j.get('evidence_verified') else ('待终态复核' if j['status'] in ACTIVE|{'pending'} else '未通过 / 未提供')
        status = LABELS.get(j['status'], j['status'])
        if j.get('process') and j['process']['matched'] is not True:
            status += '；进程未确认'
        title = (f'<a href="#log-{escape(j["name"], quote=True)}">{escape(j["name"])}</a>'
                 if normal else link(Path(j['output'])/'run.log', j['name'], study))
        rows.append(f'<tr><td>{title}</td><td>{escape(status)}</td><td>{number(depth)}</td>'
                    f'<td>{number(force,7)}</td><td>{escape(verified)}</td></tr>')
    coordinate = '法向 q' if normal else '实验室下压 d'
    return ('<div class="table"><table><thead><tr><th>案例 / 原始日志</th><th>队列状态</th>'
            f'<th>已接受 {coordinate} / mm</th><th>Fz / N</th><th>证据状态</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div>')


def analysis_section(data, study):
    report = data['report']
    controls = [j for j in data['cases'] if j.get('tag') != 'base']
    controls_finished = len(controls) == 15 and all(j['status'] not in ACTIVE|{'pending'} for j in controls)
    failed_bases = [j['name'] for j in data['cases'] if j.get('tag')=='base'
                    and j['status'] not in ACTIVE|{'pending'} and not j.get('mechanics_completed')]
    if failed_bases:
        gate = ('主路径 '+', '.join(failed_bases)+' 已终结但未完成。当前版本不能取得完整成对资格；'
                '其已接受前缀可供分析，后续数值控制仍可定位问题，不能把它仅标为“等待验证”。')
    elif any(not c.get('passed') for c in report.get('controls', [])):
        gate = '已有数值控制未通过；其余项目仍可继续诊断，当前不能宣称联合门槛通过。'
    elif not controls_finished:
        gate = '尚待验证：15 条数值控制尚未全部结束。报告中的 false 目前不能解释为已判定失败。'
    elif report.get('joint_gate_passed'):
        gate = '联合数值门槛通过；这仍不等于实验验证或全域网格收敛。'
    elif report.get('status') != data['queue'].get('status'):
        gate = '队列与分析文件尚未同步，等待终态分析写入。'
    else:
        gate = '联合数值门槛未通过或证据不完整；不得直接推广。'
    pair_lines = []
    for p in report.get('paired_contrasts', []):
        rows = p.get('rows', [])
        if not rows:
            continue
        r = rows[-1]
        pair_lines.append('<li>'+escape(f"{p['angle']}°：已有 {len(rows)} 个匹配深度记录；"
            f"两主路径完成={p.get('complete')}，配套数值控制通过={p.get('numerical_controls_passed')}。")+'</li>')
    reviews = ''.join(' · '+link(study/path,label,study) for path,label in (
        ('progress_review/index.html','最新进展与失效分析（HTML）'),
        ('pair33_review/index.html','33°完整成对分析（HTML）')) if (study/path).is_file())
    return f'''<section><h2>当前可以得出的结论</h2>
<p>{gate}</p><p>本轮检验：在相同法向深度和法向速度下，加入切向运动是否降低法向承载，
以及切向反力对竖直力与输入功的贡献。A0 是垂直基准，B 是纯法向运动，C 是法向＋切向运动。</p>
<p>共同材料、μ=0.35、刀具和固定底面保持一致；无断裂。
新路径 v<sub>n</sub>=0.1 mm/s，目标 q=0.1 mm。不能直接当作原 v=0.2 mm/s 实验室下压的实验拟合曲线。</p>
<ul>{''.join(pair_lines) or '<li>两角度的成对路径结果尚不齐，暂不能判断角度机理或实验偏差是否改善。</li>'}</ul>
<p class="sub">数值门槛同时检查 N/T/Fz/路径反力（2%）与整个原始顶面的材料位移（2 μm）。
T 是反力的切向投影，不等于单独的库仑摩擦力。局部 P2 不是已经证明无锁定的混合元。</p>
<p>{link(study/'study_report.json','分析机器证据',study)} · <a href="analysis.html">理论与本轮试验设计（HTML）</a>{reviews}</p></section>'''


def render(data, study, refresh=False):
    q, cases = data['queue'], data['cases']
    running = [j for j in cases if j['status'] == 'running']
    done = sum(bool(j.get('mechanics_completed')) for j in cases)
    ended = sum(j['status'] not in ACTIVE|{'pending'} for j in cases)
    status = LABELS.get(q.get('status'), q.get('status', '状态文件不可读'))
    if q.get('status') in ACTIVE and data['worker']['matched'] is not True:
        status += '（仅文件声明；工作进程未确认）'
    active = running[0] if running else {}
    cards = [('队列',status), ('当前工况', active.get('name','—')),
             ('已接受法向深度 q', number(active.get('accepted_depth'))+' mm'),
             ('完整求解完成', f'{done} / {len(cases)} 项')]
    body = f'''<header><span class="pill">橡胶斜切 · 本地计算</span><h1>运行状态与证据</h1>
<p class="sub">快照时间：{escape(data['observed_jst'])}　|　{escape(study.name)}</p>
<nav><a href="#current">当前求解</a><a href="#queue">全部案例</a><a href="#logs">运行日志</a><a href="analysis.html">方法与分析</a></nav></header>
<div class="cards">'''+''.join(f'<div class="card">{escape(k)}<span class="value">{escape(v)}</span></div>' for k,v in cards)+'</div>'
    body += f'<p class="sub">已终结 {ended}/{len(cases)} 项（包括失败/超时），完成数量只计完整力学路径；终态审查另列。</p>'
    if active:
        proc = active['process']
        body += f'''<section id="current"><h2>{escape(active['name'])} 当前进展</h2>
<progress max="{active['target_mm']}" value="{active.get('accepted_depth') or 0}"></progress>
<p>已接受 <b>{number(active.get('accepted_depth'))} mm</b> / 目标 {number(active['target_mm'])} mm；
保存 {active['state_count']} 个接受态。最近日志涉及深度 <b>{number(active.get('attempted_depth'))} mm</b>；日志深度不额外计入进度。</p>
<p>最后接受态：Fz = <b>{number(active.get('fz'),8)} N</b>；残差 {number(active.get('residual'),6)} N；min J = {number(active.get('min_j'),6)}。</p>
<p class="sub">求解器 PID {active.get('solver_pid')}：{escape(proc['reason'])}。
日志更新时间：{active['log_updated']}（距快照约 {number(active.get('log_age_s'),4)} 秒）。</p>
<p>最新前缀通过求解器接受门槛；整条路径的独立终态重新组装还需等待该案例结束。</p></section>'''
        if active.get('log_age_s', 0) and active['log_age_s'] > 300:
            body += '<div class="notice">日志已超过 5 分钟未更新；仅据此不能判断是求解中、停滞还是终止，需结合进程与最终错误。</div>'
    body += analysis_section(data, study)
    body += '<section id="queue"><h2>主路径与数值控制</h2>'+job_table([j for j in cases if j.get('tag')=='base'], study)
    body += '<details><summary>展开 15 条半步、积分与富集范围控制</summary>'+job_table([j for j in cases if j.get('tag')!='base'], study)+'</details></section>'
    previous = data['predecessor']
    if previous.get('jobs'):
        body += '<section><h2>前置 0° 对照</h2><p>'+escape(
            f"状态：{LABELS.get(previous.get('status'),previous.get('status'))}；"
            f"全部完整求解={previous.get('all_cases_completed')}，终态哈希检查={previous.get('terminal_hash_check_passed')}。"
            f"结束时间：{stamp(previous.get('finished_utc'))}。")+'</p>'
        body += job_table(previous['jobs'], study, normal=False)
        body += '<p class="sub">前置对照使用实验室速度 0.2 mm/s；新 A0 使用法向速度 0.1 mm/s，二者不是同一加载历史。</p></section>'
    body += '<section id="logs"><h2>运行日志</h2><p><code>FRICTION_PTC_REJECT</code> 表示一个试步未通过，求解器仍可调整后重试；'
    body += '它不同于终止错误。试步 residual 也不同于上方最后接受态的残差。</p>'
    for j in cases:
        opened = ' open' if j['status'] == 'running' else ''
        body += f'<details id="log-{escape(j["name"], quote=True)}"{opened}><summary>{escape(j["name"])} · 最近 28 行</summary><pre>{escape(j["log_tail"] or "尚无日志")}</pre>'
        if j.get('terminal_error') or j.get('result_read_error'):
            body += '<p class="bad">'+escape(str(j.get('terminal_error') or j['result_read_error']))+'</p>'
        if j.get('stderr_tail'):
            body += '<p class="bad">求解器标准错误（需结合最终状态判断）：</p><pre>'+escape(j['stderr_tail'])+'</pre>'
        body += link(Path(j['output'])/'run.log', '原始完整日志', study)+'</details>'
    body += '<details><summary>调度日志与标准错误</summary><pre>'+escape(data['worker_log'] or '调度日志为空')+'</pre>'
    body += '<pre>'+escape(data['worker_stderr'] or '调度标准错误为空')+'</pre></details>'
    body += link(study/'queue_status.json','原始队列状态',study)+' · '+link(study/'dashboard_snapshot.json','本页结构化快照',study)+'</section>'
    update = '后台只读展示程序每 60 秒刷新此文件；页面每 60 秒重载，队列结束后展示程序自动退出。' if refresh else '这是一次静态状态快照。'
    body += f'<footer>{update} 时间戳是数据新鲜度依据；不预测完成时间。HTML 不改写求解器、冻结源码、原始日志或检查点。<br>工作进程：{escape(data["worker"]["reason"])}。</footer>'
    return shell('橡胶斜切 · 运行状态', body, refresh)


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
    intro = '<nav><a href="status.html">返回运行状态</a></nav><p class="notice">理论与方案文档的 HTML 展示。文中启动时间是历史记录；实时进展以状态页快照为准。</p>'
    atomic_write(study/'analysis.html', shell('角度力学机理与试验设计', intro+'<section class="document">'+content+'</section>'))


def write_snapshot(study, refresh=False):
    data = collect(study)
    atomic_write(study/'dashboard_snapshot.json', json.dumps(data, ensure_ascii=False, indent=2))
    atomic_write(study/'status.html', render(data, study, refresh))
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
    names = {j.get('case') for j in read_json(study/'queue_status.json').get('jobs', [])}
    if names != {'A0','B33','C33','B57','C57'}:
        parser.error('This dashboard describes the matched-normal five-path study only; adapt labels before other studies')
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
