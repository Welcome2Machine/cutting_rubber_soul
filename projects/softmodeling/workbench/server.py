"""Loopback-only workbench API; one bounded compute job, no automatic retries."""
from datetime import datetime, timezone
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import subprocess
import threading
import time
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from run_support import background_child_options, write_json
from .config import ROOT, catalog, describe, normalize_config, solver_command

STATIC = Path(__file__).with_name('static')


def read_json(path, default=None):
    try: return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError): return default


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git_revision(root):
    """Git is optional for source ZIP installations; record missing provenance."""
    try:
        result=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,
                              text=True,timeout=5,check=True)
    except (OSError,subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def confined(path, base):
    """Resolve junctions/symlinks and reject traversal before touching a file."""
    base=Path(base).resolve(); path=Path(path)
    if not path.is_absolute(): path=base/path
    path=path.resolve()
    if path!=base and base not in path.parents: raise ValueError('路径必须位于指定的本地结果目录内')
    return path


def tail(path, limit=48000):
    try:
        with Path(path).open('rb') as f:
            f.seek(max(0, f.seek(0,2)-limit)); return f.read().decode('utf-8',errors='replace')
    except OSError: return ''


def guide_html():
    """Render trusted repository Markdown as escaped readable text, no plugins."""
    lines=(ROOT/'docs/research_guide.md').read_text(encoding='utf-8').splitlines()
    chunks=[]
    for line in lines:
        if line.startswith('#'):
            level=min(len(line)-len(line.lstrip('#')),4)
            chunks.append(f'<h{level}>{escape(line.lstrip("# "))}</h{level}>')
        else: chunks.append('<div class="line">'+escape(line)+'</div>')
    return ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>橡胶斜切研究导读</title><style>body{max-width:1080px;margin:40px auto;padding:0 24px;'
        'font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;color:#203744}.line{white-space:pre-wrap;'
        'min-height:.8em;overflow-wrap:anywhere}h1,h2{color:#087e8b}a{color:#087e8b}</style>'
        '<a href="/">← 仿真工作台</a>'+''.join(chunks)).encode('utf-8')


class Jobs:
    def __init__(self, root=ROOT):
        self.root=Path(root).resolve(); self.outputs=self.root/'outputs'
        self.base=self.outputs/'workbench'; self.base.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock(); self.active=None; self.process=None; self.stop_event=threading.Event()
        self.export_active=False; self.closed=False
        # Old records are displayed, never resumed or killed by an unverified PID.
        for p in (self.base/'runs').glob('*/job.json'):
            try: p=confined(p,self.base)
            except ValueError: continue
            d=read_json(p,{})
            if d.get('status') in ('prepared','running','postprocessing'):
                d.update(status='interrupted',error='服务已重启；此记录未自动恢复，原进程存活状态未认证。')
                write_json(p,d)

    def source(self, name):
        if not isinstance(name,str): raise ValueError('source 必须为仓库相对结果路径')
        p=Path(name)
        if not p.is_absolute(): p=self.root/p
        p=confined(p,self.outputs)
        if not (p/'result.json').is_file(): raise ValueError('请选择含 result.json 的算例目录')
        report=read_json(p/'result.json',{})
        expected=len(report.get('steps',[]))
        if expected and any(not (p/'states'/f'{i:04d}.npz').is_file() for i in range(expected)):
            for parent in (p,*p.parents):
                if parent==self.outputs.parent:break
                marker=read_json(parent/'storage.archive.json',{})
                if marker.get('manifest'):
                    raise ValueError('该算例的场历史已无损归档，请先恢复：python -m tools.manage_outputs restore '
                                     f'--manifest "{self.outputs/marker["manifest"]}" --prefix "{p}"')
        return p

    def artifact_url(self, path):
        return '/files/'+confined(path,self.outputs).relative_to(self.outputs).as_posix()

    def _set(self, folder, **changes):
        with self.lock:
            data=read_json(folder/'job.json',{})
            data.update(changes,updated_at=utc_now()); write_json(folder/'job.json',data)
            return data

    def get(self, identifier):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',identifier): raise ValueError('无效任务编号')
        folder=confined(self.base/'runs'/identifier,self.base)
        job=read_json(folder/'job.json')
        if job is None: raise FileNotFoundError('任务不存在')
        result=read_json(folder/'case/result.json',{})
        if result.get('steps'):
            job.update(last_depth_mm=result['steps'][-1]['depth_mm'],accepted_states=len(result['steps']))
        if job['status']=='running':
            job['solver_status']=result.get('status','initializing')
            job['elapsed_s']=max(0,time.time()-job['started_epoch'])
        return job

    def list(self):
        rows=[]
        for p in sorted((self.base/'runs').glob('*/job.json'),reverse=True):
            try: rows.append(self.get(p.parent.name))
            except ValueError: continue
        return rows

    def launch(self, config):
        c=normalize_config(config)
        with self.lock:
            if self.closed: raise ValueError('服务正在关闭')
            if self.active is not None: raise ValueError('已有一个计算任务；请完成或停止后再启动')
            identifier=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8]
            folder=self.base/'runs'/identifier; folder.mkdir(parents=True)
            write_json(folder/'config.json',c)
            cmd=solver_command(c,folder/'case')
            job=dict(id=identifier,name=c['name'],status='prepared',created_at=utc_now(),
                started_epoch=time.time(),target_mm=c['simulation']['depth_mm'],last_depth_mm=None,
                qualified=False,case_path=(folder/'case').relative_to(self.root).as_posix(),
                bundle_url=None,movie_url=None,command=cmd,config=c,git_head=git_revision(self.root),
                total_budget_s=c['resources']['total_seconds'],error=None)
            write_json(folder/'job.json',job)
            self.active=identifier; self.stop_event.clear()
            threading.Thread(target=self._run,args=(folder,c,cmd),daemon=True).start()
            return job

    def _run(self, folder, config, cmd):
        start=time.monotonic(); forced=None
        try:
            if self.stop_event.is_set(): forced='stopped'; return
            with (folder/'worker.log').open('wb') as stream:
                with self.lock:
                    self.process=subprocess.Popen(cmd,cwd=self.root,stdout=stream,stderr=subprocess.STDOUT,
                        env={**os.environ,'PYTHONUTF8':'1'},**background_child_options())
                    proc=self.process
                    self._set(folder,status='running',pid=proc.pid)
                deadline=start+config['resources']['total_seconds']
                while proc.poll() is None:
                    if self.stop_event.is_set() or time.monotonic()>deadline:
                        forced='stopped' if self.stop_event.is_set() else 'budget_exhausted'
                        proc.terminate()
                        try: proc.wait(timeout=10)
                        except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=5)
                        break
                    self.stop_event.wait(.5)
                returncode=proc.wait()
            from tools.run_material_queue import summarize_case
            result=summarize_case(folder/'case',dict(target_mm=config['simulation']['depth_mm'],
                angle=config['geometry']['angle_deg'],mu=config['contact']['mu']),returncode,forced_status=forced)
            self._set(folder,status=result['status'],qualified=result['mechanics_completed'],
                result_summary=result,returncode=returncode,error=result.get('error'),elapsed_s=time.monotonic()-start)
            if (folder/'case/result.json').exists() and forced is None:
                final_status=result['status']; self._set(folder,status='postprocessing')
                try:
                    from .results import export_case
                    bundle=export_case(folder/'case',folder/'presentation',max_frames=config['output']['max_frames'],
                        movie=config['output']['movie'])
                    movie=bundle['metadata'].get('movie',{})
                    self._set(folder,bundle_url=self.artifact_url(folder/'presentation/result.bundle.json'),
                        movie_url=self.artifact_url(folder/'presentation/indentation.mp4') if movie.get('status')=='completed' else None,
                        movie_error=movie.get('reason') if movie.get('status')=='failed' else None)
                except Exception as exc:
                    self._set(folder,export_error=f'{type(exc).__name__}: {exc}')
                self._set(folder,status=final_status)
        except Exception as exc:
            self._set(folder,status='failed',error=f'{type(exc).__name__}: {exc}')
        finally:
            if forced: self._set(folder,status=forced,qualified=False,elapsed_s=time.monotonic()-start)
            with self.lock: self.active=None; self.process=None

    def stop(self, identifier):
        with self.lock:
            self.get(identifier)
            if self.active!=identifier: raise ValueError('此任务当前不由本服务运行')
            self.stop_event.set()
        return dict(id=identifier,status='stop_requested')

    def export(self, source, max_frames=11, movie=False):
        src=self.source(source)
        if type(max_frames) is not int or not 2<=max_frames<=41 or type(movie) is not bool:
            raise ValueError('max_frames 需为2–41的整数，movie需为布尔值')
        with self.lock:
            if self.export_active: raise ValueError('已有结果导出进行中')
            self.export_active=True
        identifier=uuid4().hex; folder=self.base/'exports'/identifier; folder.mkdir(parents=True)
        write_json(folder/'job.json',dict(id=identifier,status='postprocessing',source=str(src),created_at=utc_now()))
        def worker():
            try:
                from .results import export_case
                bundle=export_case(src,folder/'presentation',max_frames=max_frames,movie=movie)
                movie_info=bundle['metadata'].get('movie',{})
                self._set(folder,status='completed',bundle_url=self.artifact_url(folder/'presentation/result.bundle.json'),
                    movie_url=self.artifact_url(folder/'presentation/indentation.mp4') if movie_info.get('status')=='completed' else None,
                    movie_error=movie_info.get('reason') if movie_info.get('status')=='failed' else None)
            except Exception as exc: self._set(folder,status='failed',error=f'{type(exc).__name__}: {exc}')
            finally:
                with self.lock: self.export_active=False
        threading.Thread(target=worker,daemon=True).start()
        return read_json(folder/'job.json')

    def close(self):
        with self.lock: self.closed=True; self.stop_event.set()
        proc=self.process
        if proc and proc.poll() is None:
            try: proc.wait(timeout=12)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=5)


def make_server(port=8765, jobs=None):
    jobs=jobs or Jobs(); token=secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args): pass

        def reply(self, payload, code=200, content_type='application/json; charset=utf-8'):
            data=payload if isinstance(payload,bytes) else json.dumps(payload,ensure_ascii=False,allow_nan=False).encode('utf-8')
            self.send_response(code); self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.send_header('X-Frame-Options','DENY')
            self.end_headers(); self.wfile.write(data)

        def authorize(self, post=False):
            allowed={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
            if self.headers.get('Host') not in allowed: raise PermissionError('仅允许本地工作台访问')
            if post:
                origin=self.headers.get('Origin')
                if origin is not None and origin not in {'http://'+s for s in allowed}: raise PermissionError('拒绝跨站请求')
                if not secrets.compare_digest(self.headers.get('X-Workbench-Token',''),token): raise PermissionError('工作台会话已失效，请刷新页面')

        def file(self, path):
            if not path.is_file(): raise FileNotFoundError('文件不存在')
            size=path.stat().st_size; start,end=0,size-1;code=200
            range_header=self.headers.get('Range')
            if range_header:
                match=re.fullmatch(r'bytes=(\d+)-(\d*)',range_header)
                if not match: raise ValueError('不支持该 Range')
                start=int(match[1]);end=min(int(match[2]) if match[2] else size-1,size-1);code=206
                if start>end: self.reply({'error':'Range超出文件'},416);return
            self.send_response(code)
            mime=mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
            self.send_header('Content-Type',mime); self.send_header('Content-Length',str(end-start+1))
            self.send_header('Accept-Ranges','bytes');self.send_header('X-Content-Type-Options','nosniff')
            if path.suffix.lower()=='.svg': self.send_header('Content-Security-Policy','sandbox')
            if code==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            with path.open('rb') as stream:
                stream.seek(start);remaining=end-start+1
                while remaining:
                    block=stream.read(min(remaining,1024*1024))
                    if not block:break
                    self.wfile.write(block);remaining-=len(block)

        def do_GET(self): self.dispatch(False)
        def do_POST(self): self.dispatch(True)

        def dispatch(self, post):
            try:
                # Drain only bounded bodies before rejection: unread bytes can reset
                # Windows TCP sockets, obscuring a useful 403 response.
                if post:
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0<length<=2*1024*1024: raise ValueError('JSON请求大小须在2 MiB以内')
                    self.connection.settimeout(10)
                    raw_body=self.rfile.read(length)
                self.authorize(post); path=unquote(urlsplit(self.path).path)
                if post:
                    body=json.loads(raw_body)
                    if not isinstance(body,dict):raise ValueError('请求必须是JSON对象')
                    if path=='/api/validate':
                        c=normalize_config(body['config'])
                        self.reply(dict(config=c,summary=describe(c),command=solver_command(c,jobs.base/'runs/NEW/case')))
                    elif path=='/api/runs':self.reply(jobs.launch(body['config']),201)
                    elif re.fullmatch('/api/runs/[A-Za-z0-9_-]+/stop',path):self.reply(jobs.stop(path.split('/')[3]))
                    elif path=='/api/export':self.reply(jobs.export(body['source'],body.get('max_frames',11),body.get('movie',False)),202)
                    else:raise FileNotFoundError('接口不存在')
                    return
                if path=='/api/catalog':self.reply(dict(catalog(),session_token=token))
                elif path=='/api/runs':self.reply(dict(runs=jobs.list()))
                elif re.fullmatch('/api/runs/[A-Za-z0-9_-]+',path):self.reply(jobs.get(path.split('/')[3]))
                elif re.fullmatch('/api/runs/[A-Za-z0-9_-]+/log',path):
                    ident=path.split('/')[3];jobs.get(ident);f=jobs.base/'runs'/ident
                    self.reply(dict(text=tail(f/'case/run.log')+'\n'+tail(f/'worker.log',8000)))
                elif path=='/api/results':
                    rows=[]
                    for f in sorted(jobs.outputs.rglob('result.json')):
                        try: f=confined(f,jobs.outputs)
                        except ValueError: continue
                        r=read_json(f,{})
                        if not r.get('steps'):continue
                        source=confined(f.parent,jobs.outputs).relative_to(jobs.root).as_posix()
                        angle=r.get('signature',{}).get('config',{}).get('inclination_deg')
                        rows.append(dict(path=source,label=f'{source} · {angle}°',status=r.get('status'),depth_mm=r['steps'][-1].get('depth_mm')))
                    self.reply(dict(results=rows))
                elif re.fullmatch('/api/export/[a-f0-9]{32}',path):
                    value=read_json(confined(jobs.base/'exports'/path.rsplit('/',1)[1]/'job.json',jobs.base))
                    if value is None:raise FileNotFoundError('导出任务不存在')
                    self.reply(value)
                elif path=='/guide':self.reply(guide_html(),content_type='text/html; charset=utf-8')
                elif path.startswith('/files/'):
                    file=confined(path[len('/files/'):],jobs.outputs)
                    if file.suffix.lower() not in ('.json','.csv','.png','.mp4','.webm','.svg'): raise PermissionError('不提供此文件类型')
                    self.file(file)
                elif path in ('/','/index.html','/app.js','/app.css'):
                    self.file(STATIC/('index.html' if path=='/' else path[1:]))
                else:raise FileNotFoundError('页面不存在')
            except (BrokenPipeError,ConnectionResetError): pass
            except PermissionError as exc:self.reply(dict(error=str(exc)),403)
            except FileNotFoundError as exc:self.reply(dict(error=str(exc)),404)
            except (ValueError,TypeError,KeyError) as exc:self.reply(dict(error=str(exc)),400)
            except Exception as exc:self.reply(dict(error=f'{type(exc).__name__}: {exc}'),500)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.jobs=jobs
    return server
