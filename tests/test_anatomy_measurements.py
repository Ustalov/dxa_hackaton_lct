import sys,unittest,copy,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dxa.anatomy_measurements import measure,lesser_trochanter

def f(*p):return {'points':list(p),'visibility':'visible'}

class Measurements(unittest.TestCase):
    def test_topmost_axis_and_clipped_endpoint(self):
        c={'shape':[200,200],'region':'spine','spacing_yx_mm':[2,1],
           'axis_definition':'L5_to_topmost',
           'detected_vertebrae':[{'anatomical_level':'L1','center':[50,60]},
                                 {'anatomical_level':'Th12','center':[30,20]},
                                 {'anatomical_level':'L5','center':[50,90]}],
           'features':{'Th12_upper':f([20,10],[40,10]),'L1_upper':f([40,50],[60,50]),
                       'L5_lower':f([40,100],[60,100])}}
        m=measure(c)['measurements']['axis_L5_topmost']
        self.assertEqual(m['top_level'],'Th12')
        self.assertEqual(m['endpoints_px'],[[50,100],[30,10]])
        self.assertAlmostEqual(m['value'],math.degrees(math.atan2(20,180)))
        square=copy.deepcopy(c);square['axis_angle_space']='square_pixels'
        square_m=measure(square)['measurements']
        self.assertAlmostEqual(square_m['axis_L5_topmost']['value'],math.degrees(math.atan2(20,90)))
        self.assertEqual(square_m['axis_L5_topmost']['physical_value_degrees'],m['value'])
        self.assertEqual(square_m['L1_height'],measure(c)['measurements']['L1_height'])
        c['features']['Th12_upper']['points'][0][1]=-1
        m=measure(c)['measurements']['axis_L5_topmost']
        self.assertIsNone(m['value']);self.assertEqual(m['top_level'],'Th12')
        c['axis_allow_estimate']=True
        c['detected_vertebrae'][1]['corners']=[[20,-1],[40,1],[20,20],[40,20]]
        c['detected_vertebrae'][2]['corners']=[[40,80],[60,80],[40,100],[60,100]]
        m=measure(c)['measurements']['axis_L5_topmost']
        self.assertAlmostEqual(m['value'],math.degrees(math.atan2(20,200)))
        self.assertTrue(m['estimated']);self.assertEqual(m['method'],'predicted_endplates_partly_outside_frame')

    def test_inclusive_hip_thresholds_and_mirror(self):
        c={'shape':[200,200],'region':'hip','spacing_yx_mm':[1,1],'lateral_image_side':'left','features':{'greater_trochanter':f([60,30]),'ischium':f([140,169]),'lateral_femur':f([20,90])}}
        keys=['greater_trochanter_top','ischium_bottom','femur_lateral_margin']
        r=measure(c)['measurements']
        for key in keys:self.assertTrue(r[key]['passes'])
        below=copy.deepcopy(c)
        below['features']['greater_trochanter']['points'][0][1]-=.01
        below['features']['ischium']['points'][0][1]+=.01
        below['features']['lateral_femur']['points'][0][0]-=.01
        for key in keys:self.assertFalse(measure(below)['measurements'][key]['passes'])
        c['features']['greater_trochanter']['points'][0][1]=31;c['features']['ischium']['points'][0][1]=168;c['features']['lateral_femur']['points'][0][0]=21
        a=measure(c)['measurements'];mir=copy.deepcopy(c);mir['lateral_image_side']='right'
        for feature in mir['features'].values():
            for p in feature['points']:p[0]=199-p[0]
        b=measure(mir)['measurements']
        for k in ['greater_trochanter_top','ischium_bottom','femur_lateral_margin']:self.assertTrue(a[k]['passes']);self.assertEqual(a[k]['value'],b[k]['value'])

    def test_lesser_dimensions_not_reversed(self):
        bad=lesser_trochanter([0,0],[4,1],[0,2],(1,1),1)
        self.assertEqual(bad['length_mm'],2);self.assertEqual(bad['medial_protrusion_mm'],4);self.assertFalse(bad['passes_visibility_shape_rule'])
        good=lesser_trochanter([0,0],[1,2],[0,4],(1,1),1);self.assertTrue(good['passes_visibility_shape_rule'])
        opposite=lesser_trochanter([0,0],[-1,2],[0,4],(1,1),1);self.assertFalse(opposite['passes_visibility_shape_rule'])

    def test_l1_half_height_and_missing_l5(self):
        c={'shape':[100,100],'region':'spine','spacing_yx_mm':[1,1],'features':{'L1_upper':f([30,10],[50,10]),'L1_lower':f([30,30],[50,30]),'L4_lower':f([30,80],[50,80])}}
        m=measure(c)['measurements'];self.assertTrue(m['L1_top_margin']['passes']);self.assertEqual(m['L1_height']['value'],20);self.assertIsNone(m['axis_L1_L5']['value'])
        c['features']['L1_upper']['visibility']='not_visible';self.assertIsNone(measure(c)['measurements']['L1_top_margin']['value'])

    def test_angles_account_for_anisotropy(self):
        c={'shape':[200,200],'region':'spine','spacing_yx_mm':[2,1],'features':{'L1_upper':f([30,10],[50,10]),'L1_lower':f([30,30],[50,30]),'L5_lower':f([40,100],[60,100]),'L4_lower':f([30,90],[50,100])},'cobb_top':'L1','cobb_bottom':'L4'}
        m=measure(c)['measurements'];self.assertAlmostEqual(m['axis_L1_L5']['value'],math.degrees(math.atan2(10,180)));self.assertAlmostEqual(m['endplate_angle']['value'],45);self.assertFalse(m['endplate_angle']['curve_endpoints_confirmed'])

    def test_spacing_compensates_resizing(self):
        a=lesser_trochanter([10,10],[13,13],[10,17],(1.05,.6),1)
        b=lesser_trochanter([20,20],[26,26],[20,34],(.525,.3),1)
        self.assertAlmostEqual(a['length_mm'],b['length_mm']);self.assertAlmostEqual(a['medial_protrusion_mm'],b['medial_protrusion_mm'])

    def test_endplate_angle_not_folded_to_acute_and_endpoint_order(self):
        c={'shape':[300,300],'region':'spine','spacing_yx_mm':[1,1],'features':{'L1_upper':f([10,10],[20,10+10*math.sqrt(3)]),'L4_lower':f([10,150],[20,150-10*math.sqrt(3)])}}
        self.assertAlmostEqual(measure(c)['measurements']['endplate_angle']['value'],120)
        c['features']['L1_upper']['points'].reverse();self.assertAlmostEqual(measure(c)['measurements']['endplate_angle']['value'],120)

    def test_invisible_and_outside_landmarks_not_used(self):
        c={'shape':[200,200],'region':'hip','features':{'greater_trochanter':f([20,500]),'ischium':{'visibility':'not_visible','points':[[50,100]]}},'lateral_image_side':'left'}
        m=measure(c)['measurements'];self.assertIsNone(m['greater_trochanter_top']['value']);self.assertIsNone(m['ischium_bottom']['value'])

if __name__=='__main__':unittest.main()
