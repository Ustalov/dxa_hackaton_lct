import json
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import numpy as np
from PIL import Image
from dxa.prototype_geometry import aggregate, physical_protrusion, iliac_geometry, l1_margin, ct_top_rule
from dxa.prototype_render import render
from dxa.prototype import read_image, TASK_NAMES, PrototypeAnalyzer


class PrototypeTests(unittest.TestCase):
    def test_unknown_cannot_become_normal(self):
        self.assertIsNone(aggregate([False,None]))
        self.assertTrue(aggregate([True,None]))
        self.assertFalse(aggregate([False,False]))

    def test_prototype_unknown_becomes_violation_with_original_evidence(self):
        for region in ('hip','spine'):
            keys=('hip_position','hip_roi') if region=='hip' else ('spine_position','spine_axis','spine_artifact')
            for missing in (False,True):
                with self.subTest(region=region,missing=missing), tempfile.TemporaryDirectory() as folder:
                    path=Path(folder)/'image.png';Image.fromarray(np.full((80,80),60,np.uint8)).save(path)
                    analyzer=PrototypeAnalyzer();analyzer.load=Mock()
                    decisions={k:False for k in keys}
                    if missing:decisions[keys[0]]=None
                    setattr(analyzer,region,Mock(return_value=(decisions,{},{})))
                    with patch('dxa.prototype.read_image',return_value=(np.full((80,80),60,np.uint8),{'region':region})):
                        result=analyzer.analyze([path],Path(folder)/'out')
                    row=result['rows'][0];d=result['details'][0]
                    self.assertEqual(row['quality_class'],int(missing))
                    self.assertIs(d['task_decisions'][keys[0]],missing)
                    self.assertIs(d['raw_task_decisions'][keys[0]],None if missing else False)
                    self.assertFalse(d['task_decisions'][keys[1]])
                    self.assertEqual(d['forced_violation_tasks'],[keys[0]] if missing else [])
                    if missing:
                        self.assertEqual(row['violation_type'],TASK_NAMES[keys[0]])
                        self.assertIn('не удалось оценить',d['error'])

    def test_model_exception_sets_applicable_tasks_to_violation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'image.png';Image.fromarray(np.full((80,80),60,np.uint8)).save(path)
            analyzer=PrototypeAnalyzer();analyzer.load=Mock();analyzer.hip=Mock(side_effect=RuntimeError('Model unavailable'))
            with patch('dxa.prototype.read_image',return_value=(np.full((80,80),60,np.uint8),{'region':'hip'})):
                result=analyzer.analyze([path],Path(folder)/'out')
            self.assertEqual(result['rows'][0]['quality_class'],1)
            self.assertEqual(result['details'][0]['task_decisions'],{'hip_position':True,'hip_roi':True})
            self.assertIn('Model unavailable',result['details'][0]['error'])

    def test_lengths_use_organizer_spacing(self):
        fem=np.zeros((80,80),bool);fem[10:60,10:30]=True
        prot=np.zeros_like(fem);prot[20:30,30:34]=True
        m=physical_protrusion(fem|prot,prot,(0,20,40,60))
        self.assertAlmostEqual(m['depth_mm'],2.4)
        self.assertAlmostEqual(m['length_mm'],10.5)
        self.assertAlmostEqual(m['area_mm2'],40*.6*1.05)

    def test_no_invented_iliac_on_upper_components(self):
        p=np.zeros((3,120,120),np.float32)
        p[0,20:50,40:70]=1
        p[1,10:30,5:30]=1  # false bone above L2
        p[2,70:90,90:110]=1
        info,_=iliac_geometry(p,['L2','iliac_L','iliac_R'])
        self.assertFalse(info['L']['present']);self.assertTrue(info['R']['present'])

    def test_l1_margin_uses_l1_not_th12(self):
        names=['T10','T11','T12','T13','L1','L2','L3','L4','L5','L6']
        p=np.zeros((10,100,100),np.float32);p[4,10:30,30:60]=1
        self.assertFalse(l1_margin(p,names)['violation'])
        self.assertTrue(l1_margin(p[:,6:],names)['violation'])

    def test_upper15_rule_strict_boundary_and_either_landmark(self):
        names=['T10','T11','T12','T13','L1','L2','L3','L4','L5','L6']
        raw=np.ones((100,100),np.uint8)
        for height,margin,expected in [(15,15,True),(16,0,False),(0,16,False),
                                       (0,15,True),(0,0,True),(0,None,None),
                                       (15,None,None),(16,None,False)]:
            with self.subTest(height=height,margin=margin):
                p=np.zeros((10,100,100),np.float32)
                if margin is not None:p[4,margin:margin+20,50:65]=1
                p[2,:height,30:45]=1
                result=ct_top_rule(p,names,raw,spacing=(1.,.6))
                self.assertIs(result['violation'],expected)

    def test_upper15_respects_spacing_and_reports_numbering(self):
        names=['T10','T11','T12','T13','L1','L2','L3','L4','L5','L6']
        raw=np.ones((100,100),np.uint8)
        p=np.zeros((10,100,100),np.float32);p[4,15:35,30:60]=1
        self.assertTrue(ct_top_rule(p,names,raw,(1.,.6))['violation'])
        self.assertFalse(ct_top_rule(p,names,raw,(1.05,.6))['violation'])
        p[5,2:12,30:60]=1
        result=ct_top_rule(p,names,raw)
        self.assertTrue(result['numbering_inconsistent'])
        self.assertIn('проверка нумерации',result['reason'])
        self.assertFalse(result['violation'])

    def test_bone_masks_never_drawn(self):
        raw=np.full((80,80),60,np.uint8)
        with tempfile.TemporaryDirectory() as d:
            for region in ('hip','spine'):
                path=Path(d)/(region+'.png')
                render(raw,region,{}, {'bone':np.ones_like(raw,bool),'femur':np.ones_like(raw,bool)},path)
                self.assertTrue(np.all(np.array(Image.open(path))==60))

    def test_png_keeps_original_dimensions_and_spacing(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'input.png';raw=np.zeros((200,300),np.uint8);raw[:,100:200]=100
            Image.fromarray(raw).save(path)
            a,m=read_image(path)
            self.assertTrue(np.array_equal(a,raw));self.assertEqual(m['spacing'],[1.05,.6])
            self.assertEqual(m['angle_space'],'square_pixels')
            Image.fromarray(np.zeros((100,100),np.uint8)).save(path)
            with self.assertRaises(ValueError):read_image(path)

    def test_region_and_side_come_from_classifier_not_width(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'input.png'
            raw=np.zeros((200,300),np.uint8);raw[:,100:200]=100
            Image.fromarray(raw).save(path)
            classifier=Mock()
            classifier.predict.return_value={'region':'hip','side':'left','score':.9,'class_name':'hip_L'}
            decoded,meta=read_image(path,region_classifier=classifier)
            self.assertTrue(np.array_equal(decoded,raw))
            self.assertEqual((meta['region'],meta['side']),('hip','left'))

    def test_dicom_decoder_allows_classifier_to_route_nonstandard_width(self):
        from pydicom.dataset import FileDataset,FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian,generate_uid
        from dxa.imaging import read_dicom
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'input.dcm';meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
            ds=FileDataset(str(path),{},file_meta=meta,preamble=b'\0'*128)
            ds.StudyInstanceUID=generate_uid();ds.SOPInstanceUID=generate_uid()
            ds.ManufacturerModelName='Lunar Prodigy';ds.Rows=64;ds.Columns=192
            ds.PhotometricInterpretation='MONOCHROME2';ds.SamplesPerPixel=1
            ds.BitsAllocated=8;ds.BitsStored=8;ds.HighBit=7;ds.PixelRepresentation=0
            raw=np.tile(np.arange(192,dtype='uint8'),(64,1));ds.PixelData=raw.tobytes();ds.save_as(path)
            with self.assertRaises(ValueError):read_dicom(path)
            classifier=Mock();classifier.predict.return_value={'region':'spine','side':'','score':.9,'class_name':'spine'}
            decoded,metadata=read_image(path,region_classifier=classifier)
            self.assertTrue(np.array_equal(decoded,raw));self.assertEqual(metadata['region'],'spine')

    def test_exact_organizer_labels(self):
        self.assertEqual(list(TASK_NAMES.values()),['Некорректная укладка','Не выровнена ось позвоночника',
                         'Присутствуют посторонние предметы','Некорректная укладка','Некорректная область интереса'])


if __name__=='__main__':unittest.main()
