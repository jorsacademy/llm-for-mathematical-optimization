import unittest
import numpy as np
import torch
from study import tasks,reward,solve,scores,TinyLM,train,CANDIDATES,evaluate
class SolverRLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_reference_reward(self):
        for t in tasks(2,8):self.assertEqual(reward(t,t['target']),1)
    def test_solver_success_not_semantics(self):
        t=tasks(2,8)[0];bad='max ge int'
        self.assertIsNotNone(solve(t,bad));self.assertEqual(reward(t,bad),0)
    def test_domain_probe(self):
        t=tasks(2,8)[0];self.assertEqual(t['target'],'min le real')
        self.assertEqual(solve(t,'min le int'),solve(t,t['target']))
        self.assertEqual(reward(t,'min le int'),0)
    def test_invalid_dsl(self):
        self.assertEqual(reward(tasks(1,1)[0],'__import__(os)'),0)
    def test_score_shape_and_gradient(self):
        m=TinyLM();s=scores(m,tasks(1,2));self.assertEqual(tuple(s.shape),(2,8))
        (-s.mean()).backward();self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters()))
    def test_rl_changes_weights(self):
        torch.manual_seed(7);m=TinyLM();before=[p.detach().clone() for p in m.parameters()]
        m,_=train(m,tasks(1,8),steps=3)
        self.assertTrue(any(not torch.equal(x,y) for x,y in zip(before,m.parameters())))
    def test_finite_evaluation(self):
        r=evaluate(TinyLM(),tasks(8,2));self.assertTrue(all(np.isfinite(x) for x in r.values()))
    def test_reproducible_tasks(self):self.assertEqual(tasks(1),tasks(1))
if __name__=='__main__':unittest.main()
