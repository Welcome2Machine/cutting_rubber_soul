"""Build an isolated FEBio + SuperLU_MT reference on Windows or Linux.

Requires CMake and C/C++ compilers (Visual Studio on Windows; GCC on Linux). Downloads
come from the official source repositories. Build products stay in .runtime.
"""
import hashlib
import argparse
import io
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile


ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'.runtime/febio'


def build_settings(system,root,jobs):
    """Portable CMake settings, without CPU-specific instruction flags."""
    if type(jobs) is not int or jobs<1:raise ValueError('Jobs must be a positive integer')
    if system=='Windows':
        return dict(superlu=root/'superlu_build/SRC/Release/superlu_mt_OPENMP.lib',
                    blas=root/'superlu_build/CBLAS/Release/blas.lib',common=[],exe='febio4.exe')
    if system!='Linux':raise RuntimeError('Supported build systems: Windows and Linux')
    return dict(superlu=root/'superlu_build/SRC/libsuperlu_mt_OPENMP.a',
                blas=root/'superlu_build/CBLAS/libblas.a',
                common=['-DCMAKE_BUILD_TYPE=Release','-DCMAKE_POSITION_INDEPENDENT_CODE=ON'],exe='febio4')


def source(repository,revision,destination,manifest):
    if (destination/'CMakeLists.txt').exists() and manifest.exists():
        return json.loads(manifest.read_text())
    url=f'https://api.github.com/repos/{repository}/zipball/{revision}'
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    data=opener.open(url,timeout=120).read()
    archive=zipfile.ZipFile(io.BytesIO(data));prefix=archive.namelist()[0].split('/')[0]
    for entry in archive.infolist():
        relative=Path(*Path(entry.filename).parts[1:]);target=(destination/relative).resolve()
        if not target.is_relative_to(destination.resolve()):raise ValueError('Unsafe source archive path')
        if entry.is_dir():target.mkdir(parents=True,exist_ok=True)
        else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(entry))
    record=dict(url=url,archive_root=prefix,sha256=hashlib.sha256(data).hexdigest())
    manifest.write_text(json.dumps(record,indent=2));return record


def main():
    global RUNTIME
    os.chdir(ROOT)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contact-audit',action='store_true',help='Build isolated opt-in contact diagnostics/correction')
    parser.add_argument('--build-only',action='store_true',help='Compile/package an already configured tree')
    parser.add_argument('--jobs',type=int,default=2,help='Parallel compiler jobs, limited by allocated CPUs and memory')
    args=parser.parse_args()
    if args.contact_audit:RUNTIME=ROOT/'.runtime/febio_audit'
    settings=build_settings(platform.system(),RUNTIME,args.jobs)
    RUNTIME.mkdir(parents=True,exist_ok=True)
    if os.name=='nt':
        vswhere=Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/Installer/vswhere.exe'
        installation=Path(subprocess.check_output([str(vswhere),'-latest','-products','*',
            '-requires','Microsoft.VisualStudio.Component.VC.Tools.x86.x64','-property','installationPath'],text=True).strip())
        cmake=installation/'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
        version=subprocess.check_output([str(vswhere),'-latest','-products','*','-property','installationVersion'],text=True).strip()
        major=int(version.split('.')[0]);generator={17:'Visual Studio 17 2022',18:'Visual Studio 18 2026'}[major]
        common=['-G',generator,'-A','x64','-DCMAKE_SUPPRESS_REGENERATION=ON']
    else:
        found=shutil.which('cmake')
        if not found:raise RuntimeError('Load a CMake module or install CMake first')
        cmake=Path(found);installation=os.environ.get('CXX','c++');generator='CMake default'
        common=settings['common']
    existing_source=(RUNTIME/'src/CMakeLists.txt').exists()
    provenance=dict(febio=source('febiosoftware/FEBio','v4.13',RUNTIME/'src',RUNTIME/'build_provenance.json'),
        superlu=source('xiaoyeli/superlu_mt','8324379',RUNTIME/'superlu_src',RUNTIME/'superlu_provenance.json'),
        compiler_installation=str(installation),generator=generator,system=platform.platform(),
        machine=platform.machine(),jobs=args.jobs)
    if args.contact_audit:
        if __package__:
            from .febio_contact_patch import apply
        else:
            from febio_contact_patch import apply
        patch_manifest=RUNTIME/'contact_patch.json'
        if patch_manifest.exists() and existing_source:
            record=json.loads(patch_manifest.read_text())
            expected={'patched_cpp_sha256':RUNTIME/'src/FEBioMech/FESlidingInterface.cpp',
                      'patched_header_sha256':RUNTIME/'src/FEBioMech/FESlidingInterface.h',
                      'recipe_sha256':ROOT/'tools/febio_contact_patch.py'}
            if any(hashlib.sha256(p.read_bytes()).hexdigest()!=record[k] for k,p in expected.items()):
                raise RuntimeError('Contact patch source/recipe differs from manifest; review before rebuilding')
            provenance['contact_patch']=record
        else:
            provenance['contact_patch']=apply(RUNTIME/'src')
            patch_manifest.write_text(json.dumps(provenance['contact_patch'],indent=2))
    # Upstream has C reference BLAS and its Makefile, but no CMakeLists here.
    # Match that Makefile by excluding the unused *myblas2.c alternatives.
    blas_cmake='file(GLOB sources "*.c")\nlist(FILTER sources EXCLUDE REGEX "myblas2\\\\.c$")\nadd_library(blas STATIC ${sources})\ntarget_include_directories(blas PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../SRC)\ntarget_compile_definitions(blas PRIVATE Add_ _CRT_SECURE_NO_WARNINGS)\n'
    (RUNTIME/'superlu_src/CBLAS/CMakeLists.txt').write_text(blas_cmake)
    provenance['cblas_cmake']=blas_cmake
    with (RUNTIME/'build_driver.log').open('wb') as log:
        def run(arguments):
            if '--build' in arguments and os.name=='nt':
                arguments=[arguments[0],arguments[1].relative_to(ROOT),*arguments[2:],'--','/nr:false','/v:minimal']
            command=[cmake.as_posix(),*[v.as_posix() if isinstance(v,Path) else str(v) for v in arguments]]
            print(' '.join(command),flush=True)
            subprocess.run(command,input=b'',stdout=log,stderr=subprocess.STDOUT,check=True)
        if not args.build_only:
            run(['-S',RUNTIME/'superlu_src','-B',RUNTIME/'superlu_build',*common,'-DTHREAD_API=OPENMP',
                '-Denable_internal_blaslib=ON','-Denable_fortran=OFF','-Denable_examples=OFF','-Denable_tests=OFF'])
            run(['--build',RUNTIME/'superlu_build','--config','Release','--target','superlu_mt_OPENMP','--parallel',args.jobs])
            run(['-S',RUNTIME/'src','-B',RUNTIME/'build',*common,'-DUSE_MKL=OFF','-DUSE_SUPERLU_MT=ON',
                f'-DSUPERLU_MT_LIB={settings["superlu"]}',
                f'-DSUPERLU_MT_INC={RUNTIME}/superlu_src/SRC',f'-DEXTRA_LIB={settings["blas"]}',
                *([] if os.name=='nt' else ['-DUSE_STATIC_STDLIBS=OFF']),
                *[f'-DUSE_{name}=OFF' for name in ('ZLIB','FFTW','MMG','LEVMAR','NLOPT','HYPRE','PDL')]])
        else:
            if not (RUNTIME/'build/CMakeCache.txt').exists():raise RuntimeError('Configure the build first')
            run(['--build',RUNTIME/'build','--config','Release','--target','febio4','--parallel',args.jobs])
    if not args.build_only:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--build-only',
                        '--jobs',str(args.jobs),*(['--contact-audit'] if args.contact_audit else [])],check=True)
        return
    binary=RUNTIME/'bin';binary.mkdir(exist_ok=True)
    for path in (RUNTIME/('build/bin/Release' if os.name=='nt' else 'build/bin')).iterdir():
        if path.suffix.lower() in ('.exe','.dll') or path.name=='febio4':
            target=binary/path.name
            if not target.exists() or hashlib.sha256(target.read_bytes()).digest()!=hashlib.sha256(path.read_bytes()).digest():
                shutil.copy2(path,target)
    if os.name!='nt':
        library=RUNTIME/'lib';library.mkdir(exist_ok=True)
        for path in (RUNTIME/'build/lib').glob('*.so*'):shutil.copy2(path,library/path.name)
        provenance['library_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in library.iterdir() if p.is_file()}
    shutil.copy2(RUNTIME/'src/LICENSE',binary/'License-FEBio.txt')
    shutil.copy2(RUNTIME/'superlu_src/License.txt',binary/'License-SuperLU_MT.txt')
    provenance['binary_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in binary.iterdir()}
    if args.contact_audit:provenance['contact_patch']=json.loads((RUNTIME/'contact_patch.json').read_text())
    (RUNTIME/'manifest.json').write_text(json.dumps(provenance,indent=2))
    print('Built',binary/settings['exe'],flush=True)


if __name__=='__main__':main()
