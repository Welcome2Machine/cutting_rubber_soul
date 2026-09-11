import tempfile
import json
import subprocess
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import numpy as np
from prototype.febio_reference import export,validate_patch,boundary_penetration_bounds,plane_strain_hex_forces,contact_log_audit,default_executable,solver_environment,RecordTail,ObliquePathAudit,run_guarded,solver_log_text
from prototype.model_config import FEMConfig,local_to_lab,_rotation_vectors
from prototype.initiation import InitiationOptions,rounded_blade_distance


class FEBioReferenceTests(unittest.TestCase):
    def test_balanced_profile_changes_numerics_but_preserves_material_and_force_gate(self):
        c=FEMConfig();o=InitiationOptions()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.feb'
            meta=export(path,c,o,case='oblique',depth=.025,speed_profile='balanced',spacing=.5)
            root=ET.parse(path).getroot()
            self.assertEqual(meta['surface_spacing_mm'],.0025)
            self.assertEqual(root.findtext('Control/time_steps'),'10')
            self.assertEqual(root.findtext('Control/solver/qn_method/max_ups'),'0')
            self.assertEqual(root.findtext('Contact/contact/tolerance'),'1e-05')
            self.assertEqual(meta['config']['newton_tolerance_n'],c.newton_tolerance_n)
            self.assertEqual(meta['config']['contact_multiplier_tolerance_n_mm2'],c.contact_multiplier_tolerance_n_mm2)
            self.assertEqual(meta['config']['contact_penetration_tolerance_mm'],2e-4)
            self.assertEqual(c.contact_penetration_tolerance_mm,1e-4)
            for name,value in c.to_dict().items():
                if name!='contact_penetration_tolerance_mm':self.assertEqual(meta['config'][name],value)
            self.assertIsNotNone(root.find('Output/logfile/node_data'))
            self.assertIsNotNone(root.find('Output/logfile/rigid_body_data'))
            self.assertIsNone(root.find('Output/logfile/element_data'))
            self.assertEqual(root.findtext('Control/plot_level'),'PLOT_FINAL')

    @unittest.skipUnless(default_executable('corrected').exists(),'Optional corrected FEBio binary absent')
    def test_balanced_actual_patch_keeps_independent_force_accuracy(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'patch'
            subprocess.run([sys.executable,'-m','prototype.febio_reference','--case','patch',
                '--speed-profile','balanced','--contact-variant','corrected','--output',str(out),
                '--execute','--timeout','30'],check=True,capture_output=True)
            self.assertTrue(json.loads((out/'status.json').read_text())['patch_verified'])

    def test_audit_cache_does_not_reuse_deformed_geometry(self):
        from prototype.febio_reference import audit_oblique_state
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.feb';meta=export(path,FEMConfig(),InitiationOptions(),case='oblique',spacing=1.)
            root=ET.parse(path).getroot();count=meta['body_nodes'];cache={}
            for step,dx in enumerate((0.,.001)):
                data=np.zeros((count,10));data[:,0]=np.arange(1,count+1);data[:,4]=dx
                tool=np.array([[2,0,0,0,0,0,0.]])
                expected=audit_oblique_state(root,meta,(step,0.,data),tool)
                actual=audit_oblique_state(root,meta,(step,0.,data),tool,cache=cache)
                self.assertEqual(actual,expected)
            self.assertEqual(len(cache),4)

    def test_fine_mesh_covers_lab_downward_motion_and_failed_material_edge(self):
        c=FEMConfig();o=InitiationOptions();h=.00125
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.feb'
            meta=export(path,c,o,case='oblique',depth=.2,spacing=.5,surface_spacing=h)
            root=ET.parse(path).getroot()
            xy=np.array([np.fromstring(v.text,sep=',')[:2] for v in root.findall('Mesh/Nodes/node')])[:meta['body_nodes']//2]
            # Invert the actual coordinate map, not the mesh-window formula.
            basis=local_to_lab(np.eye(2),c)
            material=xy@np.linalg.inv(basis)
            top=material[np.isclose(material[:,1],c.height_mm),0];top.sort()
            origin=local_to_lab(np.array([[c.drop_s_mm,c.height_mm]]),c)[0]
            trajectory=(origin+np.column_stack((np.zeros(31),-np.linspace(0,.2,31))))@np.linalg.inv(basis)
            for s in [*trajectory[:,0],7.127125,7.13168125]:
                j=np.searchsorted(top,s)
                self.assertGreater(j,0);self.assertLess(j,len(top))
                self.assertLessEqual(top[j]-top[j-1],h*(1+1e-9))
            self.assertAlmostEqual(trajectory[-1,0],7+.2*np.sin(np.pi/3))
            self.assertEqual(meta['config'],c.to_dict())
            self.assertEqual(meta['options']['tip_radius_mm'],.02)

    def test_finer_blade_reduces_chord_error_without_changing_physical_radius(self):
        c=FEMConfig();o=InitiationOptions()
        with tempfile.TemporaryDirectory() as folder:
            old=export(Path(folder)/'old.feb',c,o,case='oblique',depth=.01,spacing=1.,blade_arc_segments=64)
            new=export(Path(folder)/'new.feb',c,o,case='oblique',depth=.01,spacing=1.,blade_arc_segments=256)
            self.assertLess(new['blade_faceting_error_mm'],old['blade_faceting_error_mm']/15)
            self.assertLess(new['blade_faceting_error_mm'],c.contact_penetration_tolerance_mm/100)
            self.assertEqual(new['options'],old['options']);self.assertEqual(new['config'],old['config'])
            # At twice the refined reference spacing, a circular chord still
            # has geometric budget left. This is not a bound on actual strain.
            r=o.tip_radius_mm;length=2*.00125
            sagitta=r-np.sqrt(r*r-length*length/4)
            self.assertLess(sagitta+new['blade_faceting_error_mm'],c.contact_penetration_tolerance_mm/2)
            for invalid in (0,1.5,True):
                with self.assertRaises(ValueError):export(Path(folder)/'bad.feb',c,o,blade_arc_segments=invalid)

    def test_log_selection_preserves_accepted_event_when_file_buffer_is_truncated(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            first='augmentation # 5\nCONTACT_AUDIT t=.874038461538 aug=5 dn=1e-7 dt=1e-12 slip=.003\nconverged at time : .874038\n'
            second='augmentation # 5\nCONTACT_AUDIT t=.886538461538 aug=5 dn=7.128e-6 dt=4.66e-13 slip=.003\nconverged at time : .886538\n'
            (out/'model.log').write_text(first+second[:-35])
            (out/'stdout.log').write_text(first+second+'beginning next step\n')
            text,source=solver_log_text(out);audit=contact_log_audit(text)
            self.assertEqual(source,'stdout.log');self.assertEqual(len(audit['accepted_steps']),2)
            self.assertEqual(audit['accepted_steps'][-1]['time_s'],.886538461538)

    def test_buffered_history_and_malformed_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'nodes.txt';tail=RecordTail(path,2,3)
            path.write_bytes(b'*Step = 0\n*Time = 0\n*Data = x;y\n1 0 0\n2 0')
            self.assertEqual(tail.poll(),[])
            with path.open('ab') as stream:stream.write(b' 0\n')
            result=tail.poll();self.assertEqual(len(result),1)
            np.testing.assert_array_equal(result[0][2],[[1,0,0],[2,0,0]])
            self.assertEqual(tail.poll(final=True),[])
            with path.open('ab') as stream:stream.write(b'*Step = 1\n*Time = 1\n1 0 0\n')
            with self.assertRaisesRegex(ValueError,'Incomplete'):tail.poll(final=True)

    def test_path_failure_survives_valid_final_state_and_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'model.feb').write_text('<febio_spec/>')
            for name,row in [('nodes.txt','1 0 0 0 0 0 0 0 0 0'),('cutter.txt','2 0 0 0 0 0 0')]:
                (out/name).write_text(''.join(f'*Step = {i}\n*Time = {i}\n{row}\n' for i in range(3)))
            def state(root,metadata,node,tool,cache=None):
                return dict(step=node[0],time_s=node[1],last_depth_mm=node[1],
                    geometry_passed=node[0]!=1,necessary_force_checks_passed=True,
                    whole_boundary_penetration_upper_mm=.01 if node[0]==1 else 0.,min_corner_J=1.)
            with patch('prototype.febio_reference.audit_oblique_state',side_effect=state):
                audit=ObliquePathAudit(out,{'body_nodes':1});audit.poll(final=True)
                self.assertFalse(audit.summary['geometry_passed'])
                self.assertEqual(audit.summary['first_failed_depth_mm'],1.)
                audit.verify_coverage([{'time_s':1},{'time_s':2}])
                with self.assertRaisesRegex(ValueError,'every accepted'):audit.verify_coverage([{'time_s':2}])
                path=out/'cutter.txt';path.write_text(path.read_text().replace('*Time = 2','*Time = 3'))
                with self.assertRaisesRegex(ValueError,'different times'):
                    ObliquePathAudit(out,{'body_nodes':1}).poll(final=True)

    def test_guard_terminates_only_owned_child_on_invalid_state(self):
        from unittest.mock import Mock
        audit=Mock();audit.poll.return_value=[]
        audit.summary={'geometry_passed':False,'first_failed_depth_mm':.2}
        process=Mock();process.poll.return_value=None;process.pid=123
        with tempfile.TemporaryDirectory() as folder, patch('prototype.febio_reference.subprocess.Popen',return_value=process):
            with self.assertRaisesRegex(RuntimeError,'0.2'):
                run_guarded(Path('solver'),Path(folder),{},None,audit,{})
        process.terminate.assert_called_once();process.wait.assert_called_once()

    def test_contact_time_rounding_at_large_depth_does_not_drop_projection(self):
        log='augmentation # 3\nCONTACT_AUDIT t=12.3456789123 aug=3 dn=1e-7 dt=1e-8 slip=.001\nconverged at time : 12.3457\n'
        row=contact_log_audit(log)['accepted_steps'][0]
        self.assertEqual(row['normal_projection_mpa'],1e-7)
        self.assertEqual(row['time_s'],12.3456789123)

    def test_linux_build_paths_and_position_independent_static_dependencies(self):
        from tools.build_febio import build_settings
        root=Path('/build/febio')
        linux=build_settings('Linux',root,8)
        self.assertEqual(linux['exe'],'febio4')
        self.assertEqual(linux['superlu'].name,'libsuperlu_mt_OPENMP.a')
        self.assertEqual(linux['blas'].name,'libblas.a')
        self.assertIn('-DCMAKE_POSITION_INDEPENDENT_CODE=ON',linux['common'])
        self.assertFalse(any('native' in flag or 'x64' in flag for flag in linux['common']))
        self.assertEqual(build_settings('Windows',root,2)['exe'],'febio4.exe')
        with self.assertRaises(ValueError):build_settings('Linux',root,0)

    def test_thread_allocation_reaches_xml_and_process_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.feb'
            meta=export(path,FEMConfig(),InitiationOptions(),threads=8)
            self.assertEqual(ET.parse(path).findtext('Control/solver/linear_solver/nprocs'),'8')
            self.assertEqual(meta['threads'],8)
            env=solver_environment(8,default_executable('corrected'))
            self.assertEqual(env['OMP_NUM_THREADS'],'8')
            self.assertEqual(env['OPENBLAS_NUM_THREADS'],'1')
            for invalid in (0,-1,1.5,True):
                with self.assertRaises(ValueError):export(path,FEMConfig(),InitiationOptions(),threads=invalid)

    @unittest.skipUnless(default_executable('corrected').exists(),'Optional corrected FEBio binary absent')
    def test_corrected_actual_tangent_patch_and_strict_augmentation_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'patch'
            subprocess.run([sys.executable,'-m','prototype.febio_reference','--case','patch',
                '--contact-variant','corrected','--output',str(out),'--execute','--timeout','30'],
                check=True,capture_output=True)
            audit=json.loads((out/'contact_audit.json').read_text())
            self.assertTrue(audit['tangent_checks'])
            self.assertLess(max(t['relative'] for t in audit['tangent_checks']),1e-5)
            status=json.loads((out/'status.json').read_text())
            self.assertTrue(status['patch_verified'])
            self.assertTrue(status['accepted_nodal_contact_checks_passed'])
            failed=Path(folder)/'failed';failed.mkdir()
            path=failed/'model.feb'
            export(path,FEMConfig(),InitiationOptions(),contact_variant='corrected')
            root=ET.parse(path);root.find('Contact/contact/maxaug').text='1'
            root.write(path,encoding='utf-8',xml_declaration=True)
            exe=default_executable('corrected').resolve()
            subprocess.run([str(exe),'-i','model.feb','-nosplash'],cwd=failed,timeout=30,capture_output=True,env=solver_environment(2,exe))
            log=(failed/'model.log').read_text(errors='replace')
            self.assertNotIn('N O R M A L   T E R M I N A T I O N',log)
            self.assertEqual(contact_log_audit(log)['accepted_steps'],[])

    def test_contact_audit_does_not_confuse_rejected_and_accepted_augmentations(self):
        log='''augmentation # 51
CONTACT_AUDIT t=0.1 aug=51 dn=0.01 dt=0.2 slip=0.003
Retrying time step. Retry attempt 1 of max 12
augmentation # 7
CONTACT_AUDIT t=0.05 aug=7 dn=1e-8 dt=2e-7 slip=0.001
------- converged at time : 0.05
'''
        audit=contact_log_audit(log)
        self.assertEqual(audit['max_accepted_augmentation_index'],7)
        self.assertEqual(len(audit['accepted_steps']),1)
        self.assertEqual(audit['accepted_steps'][0]['tangential_projection_mpa'],2e-7)

    def test_corrected_contact_is_opt_in_and_mesh_replay_preserves_geometry(self):
        c=FEMConfig();o=InitiationOptions()
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'full.feb';b=Path(folder)/'prefix.feb'
            export(a,c,o,case='oblique',depth=.05,spacing=.5,surface_spacing=.0025)
            meta=export(b,c,o,case='oblique',depth=.0275,mesh_depth=.05,spacing=.5,
                        surface_spacing=.0025,contact_variant='corrected')
            full=ET.parse(a).getroot();prefix=ET.parse(b).getroot()
            self.assertEqual(ET.tostring(full.find('Mesh')),ET.tostring(prefix.find('Mesh')))
            for field in ('contact_audit','consistent_tangent','strict_augmentation'):
                self.assertIsNone(full.find('Contact/contact/'+field))
                self.assertEqual(prefix.findtext('Contact/contact/'+field),'1')
            self.assertEqual(float(prefix.findtext('Contact/contact/fric_penalty')),250.)
            self.assertEqual(prefix.findtext('Contact/contact/tolerance'),'1e-06')
            self.assertEqual(meta['mesh_target_depth_mm'],.05)

    def test_independent_hex_force_matches_material_and_nonuniform_energy_gradient(self):
        from prototype.material import response
        c=FEMConfig();ref=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
        quads=np.array([[0,1,2,3]]);f=np.array([[1.1,.4],[.1,.8]])
        current=ref@f.T+np.array([3.,-2.])
        force,energy=plane_strain_hex_forces(ref,current,quads,c)
        raw=response(f[None,:,:],c,tangent=False)
        integrated_gradient=np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])
        np.testing.assert_allclose(force,c.width_mm*integrated_gradient@raw['P'][0].T,atol=1e-12)
        self.assertAlmostEqual(energy,float(raw['energy'][0]*c.width_mm),places=12)
        current[2]+=[.13,-.07]
        force,energy=plane_strain_hex_forces(ref,current,quads,c)
        direction=np.random.default_rng(9).normal(size=current.shape);eps=1e-6
        plus=plane_strain_hex_forces(ref,current+eps*direction,quads,c)[1]
        minus=plane_strain_hex_forces(ref,current-eps*direction,quads,c)[1]
        self.assertAlmostEqual(float(np.sum(force*direction)),(plus-minus)/(2*eps),places=6)
        np.testing.assert_allclose(force.sum(axis=0),0,atol=1e-12)
        self.assertAlmostEqual(float(np.sum(current[:,0]*force[:,1]-current[:,1]*force[:,0])),0,places=11)

    def test_whole_edge_audit_detects_cutting_between_separated_nodes(self):
        c=FEMConfig();o=InitiationOptions();_,normal=_rotation_vectors(c.inclination_deg)
        center=local_to_lab(np.array([[c.drop_s_mm,c.height_mm]]),c)[0]+o.tip_radius_mm*normal
        a=center+np.array([[-2*o.tip_radius_mm,-.5*o.tip_radius_mm]])
        b=center+np.array([[2*o.tip_radius_mm,-.5*o.tip_radius_mm]])
        self.assertTrue(np.all(rounded_blade_distance(np.vstack((a,b)),c,o,0)[0]<0))
        low,high=boundary_penetration_bounds(a,b,c,o,0)
        self.assertAlmostEqual(low,.5*o.tip_radius_mm,places=10)
        self.assertGreaterEqual(high,low)
        self.assertLessEqual(high-low,1.0001e-6)

    @unittest.skipUnless(default_executable('upstream').exists(),'Optional isolated FEBio binary is absent')
    def test_real_solver_patch_and_reject_false_friction_success(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'patch'
            subprocess.run([sys.executable,'-m','prototype.febio_reference','--case','patch',
                '--output',str(out),'--execute','--timeout','30'],check=True,capture_output=True)
            metadata=json.loads((out/'config.json').read_text())
            self.assertTrue((out/'restart.dmp').is_file())
            self.assertTrue(validate_patch(out,metadata)['all_passed'])
            # A normally terminated compression calculation with omitted
            # friction must fail physical validation, even if Fy is correct.
            path=out/'cutter.txt';lines=path.read_text().splitlines()
            for i,line in enumerate(lines):
                if line and not line.startswith('*'):
                    values=line.split();values[4]='0';lines[i]=' '.join(values)
            path.write_text('\n'.join(lines)+'\n')
            self.assertFalse(validate_patch(out,metadata)['all_passed'])

    def test_geometry_plane_strain_material_and_no_hidden_contact_relaxation(self):
        c=FEMConfig();o=InitiationOptions()
        for case,surface in (('patch',None),('oblique',None),('oblique',.0025)):
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'model.feb';meta=export(path,c,o,case=case,spacing=.5,
                    surface_spacing=surface,depth=.05 if surface else None)
                root=ET.parse(path).getroot()
                nodes=np.array([np.fromstring(node.text,sep=',') for node in root.findall('Mesh/Nodes/node')])
                self.assertAlmostEqual(np.ptp(nodes[:,2]),20.)
                for part in root.findall('Mesh/Elements'):
                    for element in part:
                        ids=np.fromstring(element.text,sep=',',dtype=int)-1;x=nodes[ids]
                        a,b,d=(x[1]-x[0],x[2 if part.attrib['type']=='penta6' else 3]-x[0],x[len(ids)//2]-x[0])
                        self.assertGreater(np.linalg.det(np.stack((a,b,d))),0.)
                material=root.find('Material/material')
                self.assertEqual(float(material.findtext('c1')),c.shear_modulus_n_mm2/2)
                self.assertEqual(float(material.findtext('k')),c.bulk_modulus_n_mm2)
                self.assertEqual(material.findtext('pressure_model'),'2')
                self.assertEqual(root.find('MeshDomains/SolidDomain').attrib['three_field'],'true')
                contact=root.find('Contact/contact')
                self.assertEqual(contact.findtext('node_reloc'),'0')
                self.assertEqual(contact.attrib['type'],'sliding-node-on-facet')
                self.assertEqual(float(contact.findtext('fric_penalty')),1.)
                self.assertEqual(float(contact.findtext('fric_coeff')),c.friction_coefficient)
                constraints=root.findall("Boundary/bc[@type='linear constraint']")
                self.assertGreater(len(constraints),0)
                for item in constraints:
                    a=int(item.findtext('node'))-1;b=int(item.findtext('child_dof/node'))-1
                    np.testing.assert_allclose(nodes[a,:2],nodes[b,:2])
                    self.assertAlmostEqual(nodes[a,2]-nodes[b,2],20.)
                if case=='oblique':
                    timer=root.find('Control/time_stepper')
                    self.assertEqual(timer.findtext('dtforce'),'1')
                    self.assertEqual(timer.findtext('max_retries'),'12')
                    surface=root.find("Mesh/Surface[@name='cutter_surface']")
                    ids=np.unique(np.concatenate([np.fromstring(e.text,sep=',',dtype=int)-1 for e in surface]))
                    gaps,_=rounded_blade_distance(nodes[ids,:2],c,o,0.)
                    self.assertLess(np.max(abs(gaps)),1e-12)
                    self.assertLess(meta['blade_faceting_error_mm'],c.contact_penetration_tolerance_mm/10)


if __name__=='__main__':unittest.main()
