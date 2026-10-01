import copy
import tempfile
import unittest
from pathlib import Path
from study import production_spec,compile_spec,solve,reference,table
class IRTests(unittest.TestCase):
    def test_reference(self):
        for T in (1,2,3,6):self.assertAlmostEqual(solve(production_spec(T))['objective'],reference(T))
    def test_missing_carry(self):
        s=production_spec(3);correct=solve(s)['objective'];s['constraints'][0]['terms'].pop()
        self.assertNotAlmostEqual(solve(s)['objective'],correct)
    def test_units(self):
        s=production_spec();s['parameters']['price']['unit']={'item':1}
        with self.assertRaises(ValueError):compile_spec(s)
    def test_unknown_variable(self):
        s=production_spec();s['constraints'][0]['terms'][0]['var']='missing'
        with self.assertRaises(ValueError):compile_spec(s)
    def test_boundary(self):
        s=production_spec();del s['constraints'][0]['terms'][2]['omit_at_boundary']
        with self.assertRaises(ValueError):compile_spec(s)
    def test_missing_cell(self):
        s=production_spec();del s['parameters']['demand']['values']['0']
        with self.assertRaises(ValueError):compile_spec(s)
    def test_csv_lineage(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'d.csv';p.write_text('T,value\n0,2\n1,3\n2,2\n')
            s=production_spec();s['parameters']['demand']=table(p,['T'],{'item':1})
            r=solve(s);self.assertTrue(any(x['source'] and x['source']['row']==2 for x in r['lineage']))
    def test_duplicates(self):
        s=production_spec();s['constraints'].append(copy.deepcopy(s['constraints'][0]))
        with self.assertRaises(ValueError):compile_spec(s)
    def test_integer(self):
        s=production_spec();s['variables']['make']['kind']='integer'
        self.assertLess(solve(s)['max_violation'],1e-6)
if __name__=='__main__':unittest.main()
