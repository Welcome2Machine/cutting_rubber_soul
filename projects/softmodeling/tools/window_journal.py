"""Durable window evidence, separate from qualified results or restart authority."""
import hashlib,json,os
from pathlib import Path
from .frozen_scene import digest


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


class Journal:
    def __init__(self,directory,header):
        self.directory=Path(directory);self.path=self.directory/'window.journal.jsonl'
        self.previous=None;self.sequence=0
        with self.path.open('xb') as stream:stream.flush();os.fsync(stream.fileno())
        fd=os.open(self.directory,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        self.append('header',header)

    def append(self,kind,payload):
        row=dict(sequence=self.sequence,previous_sha256=self.previous,kind=kind,payload=payload)
        checksum=hashlib.sha256(encoded(row)).hexdigest()
        # The referenced file must be flushed before the durable accepted/rejected record.
        record=payload.get('record',{})
        for name,key in (('state_file','state_sha256'),('trial_file','trial_sha256')):
            if name in record:
                file=self.directory/record[name]
                if file.parent.resolve()!=self.directory.resolve() or digest(file)!=record[key]:raise ValueError('Unbound journal state')
                with file.open('rb') as stream:os.fsync(stream.fileno())
        with self.path.open('ab') as stream:
            stream.write(encoded(dict(**row,sha256=checksum))+b'\n');stream.flush();os.fsync(stream.fileno())
        self.previous=checksum;self.sequence+=1


def inspect(directory):
    """Verify committed evidence only. A partial tail is never silently discarded."""
    directory=Path(directory);previous=None;rows=[];terminal=False
    with (directory/'window.journal.jsonl').open('rb') as stream:
        for index,line in enumerate(stream):
            if not line.endswith(b'\n'):raise ValueError('Incomplete journal tail; retained prefix needs explicit forensic review')
            row=json.loads(line);checksum=row.pop('sha256')
            if row['sequence']!=index or row['previous_sha256']!=previous or hashlib.sha256(encoded(row)).hexdigest()!=checksum:raise ValueError('Journal chain mismatch')
            if terminal or (index==0)!=(row['kind']=='header'):raise ValueError('Invalid journal ordering')
            terminal=row['kind']=='terminal'
            for name,key in (('state_file','state_sha256'),('trial_file','trial_sha256')):
                rec=row['payload'].get('record',{})
                if name in rec:
                    file=directory/rec[name]
                    if file.parent.resolve()!=directory.resolve() or digest(file)!=rec[key]:raise ValueError('Changed journal state')
            previous=checksum;rows.append(dict(**row,sha256=checksum))
    if not rows:raise ValueError('Empty journal')
    return dict(records=rows,terminal_record_present=terminal,production_result_reconstructed=False,
                controller_verified=False,event_complete=False,physical_qualification=False)
