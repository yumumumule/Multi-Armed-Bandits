import unittest
import numpy as np
from src.bandit import BernoulliBandit
from src.algorithms import EpsilonGreedy, UCB1, UniformRandom, random_argmax
from src.simulation import simulate, seed_sequence, mean_ci, wilson_interval
from experiments.hoeffding_verification import exact_probability


class EnvironmentTests(unittest.TestCase):
    def test_validation(self):
        for p in [[], [-.1], [1.1], [np.nan], [[.2,.3]]]:
            with self.assertRaises(ValueError):
                BernoulliBandit(p)

    def test_sampling_endpoints_and_frequency(self):
        rng = np.random.default_rng(1)
        bandit = BernoulliBandit((0.,1.,.3))
        self.assertEqual(bandit.sample(0,rng),0)
        self.assertEqual(bandit.sample(1,rng),1)
        samples = [bandit.sample(2,rng) for _ in range(50000)]
        self.assertLess(abs(np.mean(samples)-.3),.01)
        for arm in [-1,3,.5]:
            with self.assertRaises(ValueError):
                bandit.sample(arm,rng)

    def test_reward_table_repeatability(self):
        b = BernoulliBandit((.2,.8))
        a = b.reward_table(1000, seed_sequence(1,'reward',0))
        np.testing.assert_array_equal(a,b.reward_table(1000,seed_sequence(1,'reward',0)))
        self.assertFalse(np.array_equal(a,b.reward_table(1000,seed_sequence(1,'reward',1))))


class PolicyTests(unittest.TestCase):
    def test_mean_updates(self):
        p = UCB1(1,2)
        for reward in [1.,0.,1.]:
            p.update(np.array([0]),np.array([reward]))
        np.testing.assert_array_equal(p.counts,[[3,0]])
        self.assertAlmostEqual(p.means[0,0],2/3)
        self.assertEqual(p.means[0,1],0)

    def test_initialization_each_arm_once(self):
        rng = np.random.default_rng(20)
        for p in [UCB1(40,5), EpsilonGreedy(40,5,.1)]:
            for _ in range(5):
                a = p.select(rng.random((40,3)))
                p.update(a,np.zeros(40))
            np.testing.assert_array_equal(p.counts,np.ones((40,5)))

    def test_ucb_index_and_exploration(self):
        p = UCB1(1,2)
        with self.assertRaises(ValueError):
            p.indices()
        p.counts[:] = [[100,1]]
        p.means[:] = [[.9,0]]
        p.rounds = 101
        np.testing.assert_allclose(p.indices(),p.means+np.sqrt(2*np.log(102)/p.counts))
        self.assertEqual(p.select(np.array([[.3,.4,.5]]))[0],1)

    def test_epsilon_extremes_and_validation(self):
        rng = np.random.default_rng(4)
        for eps in [-.1,1.1,np.nan]:
            with self.assertRaises(ValueError):
                EpsilonGreedy(1,2,eps)
        draws = rng.random((10000,3))
        p = EpsilonGreedy(10000,2,0)
        p.rounds = 2
        p.means[:] = [.2,.8]
        self.assertTrue(np.all(p.select(draws)==1))
        p.epsilon = 1
        self.assertLess(abs(p.select(draws).mean()-.5),.02)

    def test_tie_randomization_and_uniform_baseline(self):
        rng = np.random.default_rng(5)
        a = random_argmax(np.ones((20000,4)),rng.random(20000))
        self.assertTrue(np.all(np.abs(np.bincount(a)/20000-.25)<.015))
        p = UniformRandom(20000,4)
        a = p.select(rng.random((20000,3)))
        self.assertTrue(np.all(np.abs(np.bincount(a)/20000-.25)<.015))


class SimulationTests(unittest.TestCase):
    def test_pathwise_regret_reward_and_repeatability(self):
        a, raw, counts = simulate([0.,1.],['ucb1'],100,5,42,'test',[10])
        a2,raw2,counts2 = simulate([0.,1.],['ucb1'],100,5,42,'test',[10])
        self.assertEqual((a,raw,counts),(a2,raw2,counts2))
        for row in raw:
            self.assertEqual(row['reward']+row['pseudo_regret'],row['t'])
        for trial in range(5):
            bad_pulls = next(r['pulls'] for r in counts if r['trial']==trial and r['arm']==0)
            endpoint = next(r for r in raw if r['trial']==trial and r['t']==100)
            self.assertEqual(endpoint['pseudo_regret'],bad_pulls)

    def test_equal_means_zero_regret_and_all_arms_optimal(self):
        _, raw, _ = simulate([.5,.5],['epsilon_0.1','random'],20,3,0,'equal')
        self.assertTrue(all(r['pseudo_regret']==0 and r['optimal_fraction']==1 for r in raw))

    def test_policy_order_does_not_change_results(self):
        a = simulate([.2,.8],['random','ucb1'],50,4,2,'order')[1]
        b = simulate([.2,.8],['ucb1','random'],50,4,2,'order')[1]
        key = lambda r:(r['policy'],r['trial'],r['t'])
        self.assertEqual(sorted(a,key=key),sorted(b,key=key))

    def test_intervals(self):
        ci = mean_ci([1,2,3])
        self.assertAlmostEqual(ci['mean'],2)
        self.assertAlmostEqual(ci['se'],1/np.sqrt(3))
        lo, hi = wilson_interval(0,100000)
        self.assertEqual(lo,0)
        self.assertGreater(hi,0)
        self.assertLess(hi,.00004)

    def test_hoeffding_exact_lattice_event(self):
        # |S/10 - .5| >= .1 includes S=4 and S=6.
        self.assertAlmostEqual(exact_probability(10,.5,.1),1-252/1024)
        for n in [10,50,100]:
            for d in [.05,.1,.2]:
                self.assertLessEqual(exact_probability(n,.5,d),min(1,2*np.exp(-2*n*d*d))+1e-12)


if __name__ == '__main__':
    unittest.main()
